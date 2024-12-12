use std::{
    collections::HashSet,
    error::Error,
    hash::{DefaultHasher, Hash, Hasher},
    io::BufRead,
    str::FromStr,
};

use chromadb::v2::{
    client::ChromaClient,
    collection::{CollectionEntries, QueryOptions},
};
use clap::{Parser, ValueEnum};
use ollama_rs::{generation::embeddings::request::GenerateEmbeddingsRequest, Ollama};
use serde_json::Map;

mod document_struct;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Ollama model to use for embedding
    #[arg(short, long, default_value = "nomic-embed-text:latest")]
    embedding: String,

    /// Name of the project being documented
    #[arg(short, long, default_value = "bevy")]
    project: String,

    /// Distance function to use for finding neighbours
    #[arg(short, long, default_value = "squared-l2")]
    distance: Distance,

    /// Force recompute of everything from scratch
    #[arg(short, long)]
    recompute: bool,

    /// Number of results to return
    #[arg(short, long, default_value_t = 10)]
    nb_results: usize,
}

#[derive(Debug, Clone, Hash, ValueEnum)]
enum Distance {
    SquaredL2,
    InnerProduct,
    Cosine,
}

impl FromStr for Distance {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "squared-l2" => Ok(Self::SquaredL2),
            "l2" => Ok(Self::SquaredL2),
            "inner-product" => Ok(Self::InnerProduct),
            "ip" => Ok(Self::InnerProduct),
            "cosine" => Ok(Self::Cosine),
            _ => Err("Invalid distance metric".to_string()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let chroma: ChromaClient = ChromaClient::new(Default::default());
    let ollama = SimpleOllama {
        ollama: Ollama::default(),
        embedding_model: args.embedding.clone(),
    };

    let mut hash = DefaultHasher::new();
    args.embedding.hash(&mut hash);
    args.distance.hash(&mut hash);
    args.project.hash(&mut hash);
    let collection_name = hash.finish().to_string();

    let mut collection_meta = Map::new();
    collection_meta.insert(
        "hnsw:space".to_string(),
        match args.distance {
            Distance::SquaredL2 => "l2",
            Distance::InnerProduct => "ip",
            Distance::Cosine => "cosine",
        }
        .into(),
    );
    let exist = chroma.get_collection(&collection_name).await.is_ok();
    let Ok(collection) = chroma
        .get_or_create_collection(&collection_name, Some(collection_meta))
        .await
    else {
        println!("Error creating collection in Chroma");
        println!("Is the database running?");
        println!("> docker run -p 8000:8000 chromadb/chroma");
        panic!();
    };

    if !exist || args.recompute {
        std::fs::create_dir_all("out")?;
        let Ok(json_string) = std::fs::read_to_string(format!("./jsons/{}.json", args.project))
        else {
            println!("Couldn't find {}.json", args.project);
            println!(
                "You should generate all jsons from rustdoc and place them in the jsons directory by running the following commands:"
            );
            println!("You can run the following command in the project you want to document:");
            println!();
            println!(
                "> RUSTDOCFLAGS=\"-Z unstable-options --output-format json\" cargo +nightly doc"
            );
            println!();
            println!(
                "then move the generated jsons from target/doc/ to the jsons directory in the rustdoc-rag project"
            );
            panic!()
        };
        let krate: rustdoc_types::Crate = serde_json::from_str(&json_string)?;

        let mut loaded_crates = vec![None; krate.external_crates.len() + 1];

        for ext_krate in &krate.external_crates {
            let Ok(json_string) =
                std::fs::read_to_string(format!("./jsons/{}.json", ext_krate.1.name))
            else {
                continue;
            };
            let krate: rustdoc_types::Crate = serde_json::from_str(&json_string)?;
            loaded_crates[*ext_krate.0 as usize] = Some((ext_krate.1.name.clone(), krate));
        }
        loaded_crates[0] = Some(("bevy".to_string(), krate));

        let mut visited = HashSet::<(usize, rustdoc_types::Id)>::new();
        start_krate(&loaded_crates, &mut visited);

        let dir = std::fs::read_dir("./out/structs")?;
        for (i, entry) in dir.enumerate() {
            if i % 100 == 0 {
                println!("{} entries processed", i);
            }
            let entry = entry.unwrap();
            let path = entry.path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            let entries = CollectionEntries {
                ids: vec![file_name],
                embeddings: Some(vec![
                    ollama.embeddings(&std::fs::read_to_string(&path)?).await?,
                ]),
                ..Default::default()
            };
            collection.upsert(entries, None).await?;
        }
    }

    let stdin = std::io::stdin();
    println!();
    println!("Enter a prompt:");
    for line in stdin.lock().lines() {
        let query = QueryOptions {
            query_embeddings: Some(vec![ollama.embeddings(&line?).await?]),
            n_results: Some(args.nb_results as usize),
            include: Some(vec!["distances"]),
            ..Default::default()
        };
        let result = collection.query(query, None).await?;
        for (i, doc) in result.ids[0].iter().enumerate() {
            let mut doc = doc.clone();
            let _ = doc.split_off(doc.len() - 3);
            println!(
                "{:02}. {:<40} {:.3}",
                i + 1,
                doc,
                result.distances.as_ref().unwrap()[0][i]
            );
        }
        println!();
        println!("Enter a prompt:");
    }

    Ok(())
}

struct SimpleOllama {
    ollama: Ollama,
    embedding_model: String,
}

impl SimpleOllama {
    async fn embeddings(&self, document: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        let request = GenerateEmbeddingsRequest::new(self.embedding_model.clone(), document.into());
        let Ok(mut res) = self.ollama.generate_embeddings(request).await else {
            println!("Error generating embeddings");
            println!("Is Ollama running?");
            panic!();
        };
        Ok(res.embeddings.remove(0))
    }
}

type CrateCatalog = [Option<(String, rustdoc_types::Crate)>];

fn start_krate(crates: &CrateCatalog, visited: &mut HashSet<(usize, rustdoc_types::Id)>) {
    let krate = &crates[0].as_ref().unwrap().1;
    item_explorer(krate.root, 0, crates, visited, 0);
}

fn item_explorer(
    id: rustdoc_types::Id,
    current_crate: usize,
    crates: &CrateCatalog,
    visited: &mut HashSet<(usize, rustdoc_types::Id)>,
    depth: u32,
) {
    if !visited.insert((current_crate, id)) {
        return;
    }
    let krate = crates[current_crate].as_ref().unwrap();
    let item = if let Some(item) = krate.1.index.get(&id) {
        item
    } else {
        krate.1.index.get(&krate.1.root).unwrap()
    };
    match &item.inner {
        rustdoc_types::ItemEnum::Module(module) => {
            module_explorer(module, current_crate, crates, visited, depth);
        }
        rustdoc_types::ItemEnum::ExternCrate { .. } => todo!(),
        rustdoc_types::ItemEnum::Use(used) => {
            let crate_name = used.source.split("::").next().unwrap();
            if crate_name == "crate" || crate_name == "super" {
                return item_explorer(used.id.unwrap(), current_crate, crates, visited, depth + 1);
            }
            for (crate_index, krate) in crates.iter().enumerate() {
                if let Some(krate) = krate {
                    if krate.0 == crate_name {
                        return item_explorer(
                            rustdoc_types::Id(u32::MAX),
                            crate_index,
                            crates,
                            visited,
                            depth + 1,
                        );
                    }
                }
            }
            return item_explorer(used.id.unwrap(), current_crate, crates, visited, depth + 1);
        }
        rustdoc_types::ItemEnum::Union(_union) => todo!(),
        rustdoc_types::ItemEnum::Struct(stru) => {
            document_struct::document_struct(item, stru, current_crate, crates);
        }
        rustdoc_types::ItemEnum::StructField(_strufi) => {}
        rustdoc_types::ItemEnum::Enum(enume) => {
            enum_explorer(enume, current_crate, crates, visited, depth);
        }
        rustdoc_types::ItemEnum::Variant(_) => {}
        rustdoc_types::ItemEnum::Function(_) => {}
        rustdoc_types::ItemEnum::Trait(_) => {}
        rustdoc_types::ItemEnum::TraitAlias(_) => todo!(),
        rustdoc_types::ItemEnum::Impl(_) => {}
        rustdoc_types::ItemEnum::TypeAlias(_) => {}
        rustdoc_types::ItemEnum::Constant { .. } => {}
        rustdoc_types::ItemEnum::Static(_) => {}
        rustdoc_types::ItemEnum::ExternType => todo!(),
        rustdoc_types::ItemEnum::Macro(_) => {}
        rustdoc_types::ItemEnum::ProcMacro(_proc_macro) => {}
        rustdoc_types::ItemEnum::Primitive(_primitive) => todo!(),
        rustdoc_types::ItemEnum::AssocConst { .. } => todo!(),
        rustdoc_types::ItemEnum::AssocType { .. } => {}
    }
}

fn module_explorer(
    module: &rustdoc_types::Module,
    current_crate: usize,
    crates: &CrateCatalog,
    visited: &mut HashSet<(usize, rustdoc_types::Id)>,
    depth: u32,
) {
    for item in &module.items {
        item_explorer(*item, current_crate, crates, visited, depth + 1);
    }
}

fn enum_explorer(
    enumeration: &rustdoc_types::Enum,
    current_crate: usize,
    crates: &CrateCatalog,
    visited: &mut HashSet<(usize, rustdoc_types::Id)>,
    depth: u32,
) {
    enumeration.variants.iter().for_each(|variant| {
        item_explorer(*variant, current_crate, crates, visited, depth + 1);
    });
}

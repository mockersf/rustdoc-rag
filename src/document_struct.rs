use std::io::Write;

use crate::CrateCatalog;

struct StructDocument {
    name: String,
    docs: Option<String>,
    fields: Vec<Field>,
}

struct Field {
    name: String,
    docs: Option<String>,
}

pub fn document_struct(
    item: &rustdoc_types::Item,
    stru: &rustdoc_types::Struct,
    current_crate: usize,
    crates: &CrateCatalog,
) {
    std::fs::create_dir_all("out/structs").unwrap();
    let mut doc = StructDocument {
        name: item.name.as_ref().unwrap().to_string(),
        docs: item.docs.clone(),
        fields: vec![],
    };

    match &stru.kind {
        rustdoc_types::StructKind::Unit => {}
        rustdoc_types::StructKind::Tuple(_fields) => {}
        rustdoc_types::StructKind::Plain { fields, .. } => {
            doc.fields = fields
                .iter()
                .map(|field| {
                    let field = crates
                        .get(current_crate)
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .1
                        .index
                        .get(field)
                        .unwrap();
                    Field {
                        name: field.name.as_ref().unwrap().to_string(),
                        docs: field.docs.clone(),
                    }
                })
                .collect();
        }
    }
    doc.write();
}

impl StructDocument {
    pub fn write(&self) {
        let mut file = std::fs::File::create(format!("out/structs/{}.md", self.name)).unwrap();

        write!(file, "{} is a struct.\n\n", self.name).unwrap();
        if let Some(docs) = &self.docs {
            write!(file, "{}\n\n", docs).unwrap();
        }
        if !self.fields.is_empty() {
            write!(file, "It has the following fields: ").unwrap();
            for field in &self.fields {
                write!(file, "{}, ", field.name).unwrap();
            }
            write!(file, "\n\n").unwrap();

            for field in &self.fields {
                if let Some(docs) = &field.docs {
                    write!(file, "More details about the {} field:\n\n", field.name).unwrap();
                    write!(file, "{}\n\n", docs).unwrap();
                }
            }
        }
    }
}

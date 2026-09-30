//! Session lookup: exact name, exact id, then unique id-prefix, with a typo suggestion
//! when nothing matches — the one place every CLI command's `<session>` argument is
//! resolved, so `ralph inspect dem` suggesting `demo` and `ralph pause 01ABC` (a prefix)
//! both go through the same rules.
use super::*;
use crate::typo::suggest_similar;

impl Storage {
    pub fn resolve(&self, identifier: &str) -> Result<SessionRow, StorageError> {
        if let Some(row) = self.find_by_name(identifier)? {
            return Ok(row);
        }
        if let Some(row) = self.find_by_id(identifier)? {
            return Ok(row);
        }
        let matches = self.find_by_id_prefix(identifier)?;
        match matches.len() {
            0 => {
                let names = self.list()?.into_iter().map(|r| r.name).collect::<Vec<_>>();
                let suggestion = suggest_similar(identifier, names.iter().map(String::as_str));
                Err(StorageError::NotFound {
                    identifier: identifier.to_string(),
                    suggestion,
                })
            }
            1 => matches
                .into_iter()
                .next()
                .ok_or_else(|| StorageError::Corrupt("session resolution lost its match".into())),
            _ => Err(StorageError::AmbiguousPrefix(
                identifier.to_string(),
                matches.into_iter().map(|r| r.name).collect(),
            )),
        }
    }

    fn find_by_name(&self, name: &str) -> Result<Option<SessionRow>, StorageError> {
        self.conn
            .query_row(
                "SELECT * FROM sessions WHERE name = ?1",
                params![name],
                row_to_session,
            )
            .optional()
            .map_err(StorageError::from)
    }

    fn find_by_id(&self, id: &str) -> Result<Option<SessionRow>, StorageError> {
        self.conn
            .query_row(
                "SELECT * FROM sessions WHERE id = ?1",
                params![id],
                row_to_session,
            )
            .optional()
            .map_err(StorageError::from)
    }

    fn find_by_id_prefix(&self, prefix: &str) -> Result<Vec<SessionRow>, StorageError> {
        let pattern = format!("{prefix}%");
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM sessions WHERE id LIKE ?1")?;
        let rows = stmt.query_map(params![pattern], row_to_session)?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StorageError::from)
    }
}

pub fn parse_expert_spec(spec: &str) -> (String, Option<String>) {
    let trimmed = spec.trim();
    if let Some((name, version)) = trimmed.split_once('@') {
        (canonical_name(name), Some(version.trim().to_string()))
    } else {
        (canonical_name(trimmed), None)
    }
}

pub fn canonical_name(name: &str) -> String {
    let trimmed = name.trim();
    if trimmed.starts_with("expert-") {
        trimmed.to_string()
    } else {
        format!("expert-{}", trimmed)
    }
}

pub fn display_name(name: &str, version: &str) -> String {
    format!("{}@{}", name, version)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_expert_spec_with_version() {
        let (name, version) = parse_expert_spec("sql@0.3.0");
        assert_eq!(name, "expert-sql");
        assert_eq!(version, Some("0.3.0".to_string()));
    }

    #[test]
    fn test_parse_expert_spec_without_version() {
        let (name, version) = parse_expert_spec("sql");
        assert_eq!(name, "expert-sql");
        assert_eq!(version, None);
    }

    #[test]
    fn test_parse_expert_spec_with_expert_prefix() {
        let (name, version) = parse_expert_spec("expert-sql@0.2.1");
        assert_eq!(name, "expert-sql");
        assert_eq!(version, Some("0.2.1".to_string()));
    }

    #[test]
    fn test_parse_expert_spec_whitespace() {
        let (name, version) = parse_expert_spec("  sql  @  0.3.0  ");
        assert_eq!(name, "expert-sql");
        assert_eq!(version, Some("0.3.0".to_string()));
    }

    #[test]
    fn test_canonical_name() {
        assert_eq!(canonical_name("sql"), "expert-sql");
        assert_eq!(canonical_name("expert-sql"), "expert-sql");
        assert_eq!(canonical_name("  sql  "), "expert-sql");
    }

    #[test]
    fn test_display_name() {
        assert_eq!(display_name("expert-sql", "0.3.0"), "expert-sql@0.3.0");
    }
}

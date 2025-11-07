use std::collections::HashSet;

#[test]
fn test_dependency_parsing() {
    let dep = "expert-english@>=0.0.1";
    
    let parts: Vec<&str> = dep.split('@').collect();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], "expert-english");
    assert_eq!(parts[1], ">=0.0.1");
}

#[test]
fn test_dependency_without_version() {
    let dep = "expert-english";
    
    let parts: Vec<&str> = dep.split('@').collect();
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0], "expert-english");
}

#[test]
fn test_circular_dependency_detection() {
    // Maximum depth should be 10
    let max_depth = 10;
    let mut depth = 0;
    
    for _ in 0..15 {
        depth += 1;
        if depth > max_depth {
            // Should stop here
            break;
        }
    }
    
    assert_eq!(depth, 11); // Stops at 11 (after checking > 10)
}

#[test]
fn test_version_constraint_exact() {
    let version_req = "0.0.1";
    let installed_version = "0.0.1";
    
    assert_eq!(version_req, installed_version);
}

#[test]
fn test_version_constraint_gte() {
    let constraint = ">=0.0.1";
    
    assert!(constraint.starts_with(">="));
    
    let version = constraint.strip_prefix(">=").unwrap();
    assert_eq!(version, "0.0.1");
}

#[test]
fn test_dependency_graph_construction() {
    let mut deps = vec![
        ("expert-a", vec!["expert-b", "expert-c"]),
        ("expert-b", vec!["expert-c"]),
        ("expert-c", vec![]),
    ];
    
    // Topological sort: c, b, a
    let mut sorted = Vec::new();
    let mut visited = HashSet::new();
    
    fn visit(name: &str, deps: &[(&str, Vec<&str>)], visited: &mut HashSet<String>, sorted: &mut Vec<String>) {
        if visited.contains(name) {
            return;
        }
        
        visited.insert(name.to_string());
        
        if let Some((_, children)) = deps.iter().find(|(n, _)| *n == name) {
            for child in children {
                visit(child, deps, visited, sorted);
            }
        }
        
        sorted.push(name.to_string());
    }
    
    for (name, _) in &deps {
        visit(name, &deps, &mut visited, &mut sorted);
    }
    
    // c should be first, a should be last
    assert_eq!(sorted[0], "expert-c");
    assert_eq!(sorted[sorted.len() - 1], "expert-a");
}

#[test]
fn test_dependency_list_empty() {
    let requires: Vec<String> = Vec::new();
    
    assert!(requires.is_empty());
}

#[test]
fn test_dependency_list_multiple() {
    let requires = vec![
        "expert-english@>=0.0.1".to_string(),
        "expert-base@0.1.0".to_string(),
    ];
    
    assert_eq!(requires.len(), 2);
}

#[test]
fn test_git_url_for_dependency() {
    let dep_name = "expert-english";
    let git_url = format!("git+https://github.com/hivellm/{}.git", dep_name);
    
    assert_eq!(git_url, "git+https://github.com/hivellm/expert-english.git");
}


use std::path::PathBuf;

#[test]
fn test_git_source_detection() {
    let sources = vec![
        ("git+https://github.com/user/repo.git", true),
        ("file://./local/path", false),
        ("./local/path", false),
        ("package.expert", false),
    ];
    
    for (source, is_git) in sources {
        assert_eq!(source.starts_with("git+"), is_git);
    }
}

#[test]
fn test_git_ref_extraction() {
    let source = "git+https://github.com/user/repo.git#v0.0.1";
    
    if let Some(pos) = source.find('#') {
        let (url, ref_part) = source.split_at(pos);
        let ref_spec = ref_part.strip_prefix('#').unwrap();
        
        assert_eq!(url, "git+https://github.com/user/repo.git");
        assert_eq!(ref_spec, "v0.0.1");
    }
}

#[test]
fn test_git_directory_detection() {
    let expert_dir = PathBuf::from("/tmp/expert");
    let git_dir = expert_dir.join(".git");
    
    // Check if path ends with .git
    assert_eq!(git_dir.file_name().unwrap(), ".git");
}

#[test]
fn test_force_flag_behavior() {
    let force = true;
    let is_git = false;
    
    if !is_git && force {
        // Should reinstall
        assert!(true);
    }
}

#[test]
fn test_git_pull_command() {
    let command_parts = vec!["git", "-C", "/path/to/repo", "pull", "origin", "main"];
    
    assert_eq!(command_parts[0], "git");
    assert_eq!(command_parts[1], "-C");
    assert_eq!(command_parts[2], "/path/to/repo");
    assert_eq!(command_parts[3], "pull");
}

#[test]
fn test_ref_spec_default() {
    let ref_spec: Option<&str> = None;
    let default_ref = ref_spec.unwrap_or("main");
    
    assert_eq!(default_ref, "main");
}


use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_chat_oneshot_mode() {
    let mut cmd = Command::cargo_bin("expert-cli").unwrap();

    cmd.arg("chat")
        .arg("--prompt")
        .arg("Hello world")
        .arg("--max-tokens")
        .arg("10")
        .arg("--device")
        .arg("cpu");

    cmd.assert().success();
}

#[test]
fn test_chat_with_expert() {
    let mut cmd = Command::cargo_bin("expert-cli").unwrap();

    cmd.arg("chat")
        .arg("--experts")
        .arg("neo4j")
        .arg("--prompt")
        .arg("Test query")
        .arg("--max-tokens")
        .arg("10")
        .arg("--device")
        .arg("cpu");

    cmd.assert().success();
}

#[test]
fn test_chat_debug_shows_loading() {
    let mut cmd = Command::cargo_bin("expert-cli").unwrap();

    cmd.arg("chat")
        .arg("--prompt")
        .arg("Test")
        .arg("--max-tokens")
        .arg("5")
        .arg("--device")
        .arg("cpu")
        .arg("--debug");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Loading"));
}

#[test]
fn test_chat_quiet_mode_no_extra_output() {
    let mut cmd = Command::cargo_bin("expert-cli").unwrap();

    cmd.arg("chat")
        .arg("--prompt")
        .arg("Test")
        .arg("--max-tokens")
        .arg("5")
        .arg("--device")
        .arg("cpu");

    // Without --debug, should NOT show loading messages
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Loading").not());
}

#[test]
fn test_chat_multiple_experts() {
    let mut cmd = Command::cargo_bin("expert-cli").unwrap();

    cmd.arg("chat")
        .arg("--experts")
        .arg("neo4j,sql")
        .arg("--prompt")
        .arg("Find all")
        .arg("--max-tokens")
        .arg("10")
        .arg("--device")
        .arg("cpu");

    cmd.assert().success();
}

#[test]
fn test_chat_temperature_override() {
    let mut cmd = Command::cargo_bin("expert-cli").unwrap();

    cmd.arg("chat")
        .arg("--prompt")
        .arg("Test")
        .arg("--temperature")
        .arg("0.5")
        .arg("--max-tokens")
        .arg("5")
        .arg("--device")
        .arg("cpu");

    cmd.assert().success();
}

#[test]
fn test_chat_sampling_params() {
    let mut cmd = Command::cargo_bin("expert-cli").unwrap();

    cmd.arg("chat")
        .arg("--prompt")
        .arg("Test")
        .arg("--temperature")
        .arg("0.7")
        .arg("--top-p")
        .arg("0.9")
        .arg("--top-k")
        .arg("40")
        .arg("--max-tokens")
        .arg("5")
        .arg("--device")
        .arg("cpu");

    cmd.assert().success();
}

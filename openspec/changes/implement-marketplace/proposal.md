# Implement Git-Based Marketplace (P2)

## Why

Enable community to discover, share, and install experts via decentralized Git-based marketplace. No centralized registry (NPM/PyPI), just a Git index of expert repositories.

## What Changes

- Create marketplace index repository
- Implement `expert-cli marketplace` commands
- Add search and discovery
- Implement publisher verification
- Create web UI for browsing
- Enable community submissions (PRs)

**Non-breaking**: Adds marketplace functionality, direct Git install still works

## Impact

- **Affected specs**: marketplace-index, cli-marketplace, search, submission
- **Affected code**: New `/expert/marketplace/` + CLI subcommands
- **Timeline**: P2 milestone (6-8 weeks after P1)
- **Community**: Enables expert ecosystem growth


#!/bin/bash
# Clean up old Unsloth compiled caches from individual expert directories
#
# This script removes the deprecated unsloth_compiled_cache directories
# from expert directories. As of now, all Unsloth compiled caches are
# centralized in expert/cache/unsloth_compiled/ to avoid duplication.
#
# Usage:
#   ./cleanup-old-caches.sh          # Delete old caches
#   ./cleanup-old-caches.sh --dry-run # Show what would be deleted

set -euo pipefail

DRY_RUN=false
VERBOSE=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERT_ROOT="$(dirname "$SCRIPT_DIR")"
EXPERTS_DIR="$EXPERT_ROOT/experts"

echo ""
echo "================================================"
echo "  Unsloth Cache Cleanup"
echo "================================================"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "\033[33m[DRY RUN MODE] No files will be deleted\033[0m"
    echo ""
fi

# Find all unsloth_compiled_cache directories
mapfile -t CACHE_DIRS < <(find "$EXPERTS_DIR" -type d -name "unsloth_compiled_cache" 2>/dev/null || true)

if [ ${#CACHE_DIRS[@]} -eq 0 ]; then
    echo -e "\033[32m[OK] No old cache directories found\033[0m"
    echo ""
    exit 0
fi

echo -e "\033[33mFound ${#CACHE_DIRS[@]} old cache director(ies):\033[0m"
echo ""

TOTAL_SIZE=0

for CACHE_DIR in "${CACHE_DIRS[@]}"; do
    # Get relative path
    REL_PATH="${CACHE_DIR#$EXPERT_ROOT/}"
    
    # Calculate size
    SIZE=$(du -sb "$CACHE_DIR" 2>/dev/null | cut -f1 || echo "0")
    SIZE_MB=$(echo "scale=2; $SIZE / 1024 / 1024" | bc)
    TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    
    echo "  - $REL_PATH"
    echo "    Size: ${SIZE_MB} MB"
    
    if [ "$VERBOSE" = true ]; then
        # Show file count
        FILE_COUNT=$(find "$CACHE_DIR" -type f | wc -l)
        echo "    Files: $FILE_COUNT"
    fi
    
    if [ "$DRY_RUN" = false ]; then
        if rm -rf "$CACHE_DIR"; then
            echo -e "    \033[31m[DELETED]\033[0m"
        else
            echo -e "    \033[31m[ERROR] Failed to delete\033[0m"
        fi
    else
        echo -e "    \033[33m[WILL BE DELETED]\033[0m"
    fi
    
    echo ""
done

TOTAL_SIZE_MB=$(echo "scale=2; $TOTAL_SIZE / 1024 / 1024" | bc)

echo "================================================"
echo "Summary:"
echo "  Total size: ${TOTAL_SIZE_MB} MB"

if [ "$DRY_RUN" = true ]; then
    echo -e "  Status: \033[33mDry run - no files deleted\033[0m"
    echo ""
    echo -e "\033[33mRun without --dry-run to actually delete the caches\033[0m"
else
    echo -e "  Status: \033[32mCleanup complete\033[0m"
    echo ""
    echo -e "\033[32mAll Unsloth compiled caches are now centralized in:\033[0m"
    echo "  expert/cache/unsloth_compiled/"
fi

echo "================================================"
echo ""


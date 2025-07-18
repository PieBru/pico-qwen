#!/bin/bash

# make_clean.sh - Comprehensive cleanup script for Qwen3 C inference engine
# Removes all build artifacts and temporary files across Rust, C/C++, and Python
#
# Usage: ./scripts/make_clean.sh [options]
# Options:
#   -h, --help     Show this help message
#   -v, --verbose  Show detailed cleanup information
#   -f, --force    Skip confirmation prompts
#   -d, --dry-run  Show what would be removed without actually deleting

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERBOSE=false
FORCE=false
DRY_RUN=false

# Help function
show_help() {
    cat << EOF
${GREEN}Qwen3 Cleanup Script${NC}

${BLUE}Usage:${NC}
    ./scripts/make_clean.sh [options]

${BLUE}Options:${NC}
    -h, --help     Show this help message
    -v, --verbose  Show detailed cleanup information
    -f, --force    Skip confirmation prompts
    -d, --dry-run  Show what would be removed without deleting

${BLUE}Description:${NC}
    Cleans all build artifacts and temporary files across:
    • Rust: target/ directories, Cargo.lock
    • C/C++: *.o, *.a, *.so, executables, build/
    • Python: __pycache__, *.pyc, *.pyo, build/, dist/
    • System: .DS_Store, .tmp, *.log, coverage files
    • Debug symbols and profiling data

${BLUE}Examples:${NC}
    ./scripts/make_clean.sh           # Interactive cleanup
    ./scripts/make_clean.sh -f        # Force cleanup without prompts
    ./scripts/make_clean.sh -v        # Verbose cleanup
    ./scripts/make_clean.sh -d        # Dry run to see what would be cleaned

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown option '$1'"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Logging functions
log() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${GREEN}[INFO]${NC} $1"
    fi
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_dry() {
    echo -e "${BLUE}[DRY-RUN]${NC} Would remove: $1"
}

# Safe removal function
safe_remove() {
    local path="$1"
    local description="$2"
    
    if [[ ! -e "$path" ]]; then
        return 0
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_dry "$description: $path"
        return 0
    fi
    
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${GREEN}[REMOVE]${NC} $description: $path"
    fi
    
    if [[ -d "$path" ]]; then
        rm -rf "$path"
    else
        rm -f "$path"
    fi
}

# Size calculation function
calculate_size() {
    local path="$1"
    if [[ -e "$path" ]]; then
        du -sh "$path" 2>/dev/null | cut -f1
    else
        echo "0B"
    fi
}

# Confirmation prompt
confirm_cleanup() {
    if [[ "$FORCE" == true ]] || [[ "$DRY_RUN" == true ]]; then
        return 0
    fi
    
    echo -e "${YELLOW}This will remove all build artifacts and temporary files.${NC}"
    read -p "Continue? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
}

# Main cleanup function
main() {
    echo -e "${GREEN}Starting Qwen3 cleanup...${NC}"
    
    # Get workspace root
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local workspace_root="$(cd "$script_dir/.." && pwd)"
    cd "$workspace_root"
    
    # Calculate total size before cleanup
    local total_size=0
    
    # File patterns to remove (safely excluding project files)
    local patterns=(
        # C/C++ build artifacts (but NOT Makefiles or source files)
        "*.o" "*.obj" "*.a" "*.lib" "*.so" "*.dylib" "*.dll"
        "*.exe" "*.out" "*.app" "a.out"
        
        # Rust build artifacts (but NOT Cargo.lock for reproducible builds)
        "target/"
        
        # Python build artifacts
        "__pycache__/" "*.pyc" "*.pyo" "*.pyd" "*.pyi"
        "build/" "dist/" "*.egg-info/" "*.egg" ".pytest_cache/" ".mypy_cache/"
        
        # CMake build (but NOT project Makefiles)
        "cmake-build-*/" "CMakeCache.txt" "CMakeFiles/" "cmake_install.cmake"
        
        # Coverage and profiling
        "*.gcda" "*.gcno" "*.gcov" "*.profraw" "*.profdata" ".coverage" "htmlcov/"
        
        # Debug symbols
        "*.dSYM/" "*.debug" "*.pdb"
        
        # Temporary files
        "*.tmp" "*.temp" "*.log" "*.bak" "*.swp" "*.swo" "*~" ".#*" "#*#"
        
        # System files
        ".DS_Store" ".DS_Store?" "._*" "Thumbs.db" "desktop.ini"
        
        # IDE files
        ".vscode/" ".idea/" "*.iml" ".vs/" "*.user" "*.suo" "*.ncb" "*.opendb"
        
        # Documentation build
        "docs/_build/" "docs/build/"
        
        # Test artifacts (but NOT test source files)
        "test_results/"
    )
    
    # Directories to remove (build artifacts only)
    local directories=(
        "build/"
        "debug/"
        "release/"
        "bin/"
        "lib/"
    )
    
    # Create list of items to clean
    local items_to_clean=()
    
    # Find files by pattern
    for pattern in "${patterns[@]}"; do
        while IFS= read -r -d '' item; do
            items_to_clean+=("$item")
        done < <(find . -name "$pattern" -print0 2>/dev/null || true)
    done
    
    # Find directories
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            items_to_clean+=("$dir")
        fi
    done
    
    # Calculate total size
    if [[ ${#items_to_clean[@]} -gt 0 ]]; then
        echo -e "${BLUE}Calculating cleanup size...${NC}"
        for item in "${items_to_clean[@]}"; do
            if [[ -e "$item" ]]; then
                local size=$(du -sh "$item" 2>/dev/null | cut -f1)
                total_size=$((total_size + $(du -sb "$item" 2>/dev/null | cut -f1)))
                log "Found: $item ($size)"
            fi
        done
    fi
    
    # Convert total size to human readable
    local total_size_human=$(du -sh . 2>/dev/null | cut -f1)
    
    if [[ ${#items_to_clean[@]} -eq 0 ]]; then
        echo -e "${GREEN}No artifacts found to clean.${NC}"
        return 0
    fi
    
    echo -e "${YELLOW}Found ${#items_to_clean[@]} items to clean${NC}"
    
    confirm_cleanup
    
    # Perform cleanup
    echo -e "${GREEN}Cleaning artifacts...${NC}"
    
    for item in "${items_to_clean[@]}"; do
        if [[ -e "$item" ]]; then
            local item_type=""
            if [[ -d "$item" ]]; then
                item_type="directory"
            else
                item_type="file"
            fi
            safe_remove "$item" "$item_type"
        fi
    done
    
    # Clean additional specific locations
    safe_remove "target/" "Rust build directory"
    safe_remove "qwen3-c-lib/build/" "C build directory"
    safe_remove "qwen3-c-lib/debug/" "C debug directory"
    safe_remove "qwen3-c-lib/release/" "C release directory"
    
    # Clean Python cache
    safe_remove "__pycache__/" "Python cache"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    
    # Clean Rust artifacts
    if [[ -d "target/" ]]; then
        safe_remove "target/" "Rust target directory"
    fi
    
    # Clean CMake artifacts
    find . -type f -name "CMakeCache.txt" -delete 2>/dev/null || true
    find . -type d -name "CMakeFiles" -exec rm -rf {} + 2>/dev/null || true
    
    echo -e "${GREEN}Cleanup completed!${NC}"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${BLUE}Dry run completed - no files were actually removed.${NC}"
    else
        echo -e "${GREEN}All artifacts have been cleaned.${NC}"
    fi
}

# Trap to handle script interruption
trap 'echo -e "\n${RED}Cleanup interrupted by user.${NC}"; exit 1' INT TERM

# Run main function
main "$@"
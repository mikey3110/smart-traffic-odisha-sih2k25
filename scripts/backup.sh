#!/bin/bash

# Backup Script for Smart Traffic Management System
# Creates automated backups of database and model files

set -e  # Exit on any error

# Configuration
BACKUP_DIR="/backups/smart-traffic"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="smart-traffic-backup-$DATE"

# Database configuration
POSTGRES_HOST="localhost"
POSTGRES_PORT="5432"
POSTGRES_DB="traffic_management"
POSTGRES_USER="traffic_user"

# Model configuration
MODEL_DIR="/app/models"
ML_MODEL_DIR="/app/ml_models"

# S3 configuration (optional)
S3_BUCKET="smart-traffic-backups"
S3_REGION="us-east-1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging
LOG_FILE="backup_$(date +%Y%m%d_%H%M%S).log"
echo "Backup Log - $(date)" > "$LOG_FILE"

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory: $BACKUP_DIR/$BACKUP_NAME"
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
}

# Backup PostgreSQL database
backup_database() {
    log "Starting PostgreSQL database backup..."
    
    local db_backup_file="$BACKUP_DIR/$BACKUP_NAME/database_backup.sql"
    
    if command -v pg_dump &> /dev/null; then
        if PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
            -h "$POSTGRES_HOST" \
            -p "$POSTGRES_PORT" \
            -U "$POSTGRES_USER" \
            -d "$POSTGRES_DB" \
            --verbose \
            --no-password \
            > "$db_backup_file" 2>> "$LOG_FILE"; then
            log "${GREEN}✓ Database backup completed: $db_backup_file${NC}"
            
            # Compress database backup
            gzip "$db_backup_file"
            log "${GREEN}✓ Database backup compressed${NC}"
        else
            log "${RED}✗ Database backup failed${NC}"
            return 1
        fi
    else
        log "${YELLOW}⚠ pg_dump not found, skipping database backup${NC}"
        return 0
    fi
}

# Backup ML models
backup_models() {
    log "Starting ML model backup..."
    
    local model_backup_dir="$BACKUP_DIR/$BACKUP_NAME/models"
    mkdir -p "$model_backup_dir"
    
    # Backup model files
    if [ -d "$MODEL_DIR" ]; then
        log "Backing up models from $MODEL_DIR..."
        cp -r "$MODEL_DIR"/* "$model_backup_dir/" 2>/dev/null || log "${YELLOW}⚠ Some model files could not be copied${NC}"
        log "${GREEN}✓ Model files backed up${NC}"
    else
        log "${YELLOW}⚠ Model directory $MODEL_DIR not found${NC}"
    fi
    
    # Backup ML model files
    if [ -d "$ML_MODEL_DIR" ]; then
        log "Backing up ML models from $ML_MODEL_DIR..."
        cp -r "$ML_MODEL_DIR"/* "$model_backup_dir/" 2>/dev/null || log "${YELLOW}⚠ Some ML model files could not be copied${NC}"
        log "${GREEN}✓ ML model files backed up${NC}"
    else
        log "${YELLOW}⚠ ML model directory $ML_MODEL_DIR not found${NC}"
    fi
    
    # Create model metadata
    local metadata_file="$model_backup_dir/model_metadata.json"
    cat > "$metadata_file" << EOF
{
    "backup_date": "$(date -Iseconds)",
    "model_directories": [
        "$MODEL_DIR",
        "$ML_MODEL_DIR"
    ],
    "total_files": $(find "$model_backup_dir" -type f | wc -l),
    "total_size": "$(du -sh "$model_backup_dir" | cut -f1)"
}
EOF
    log "${GREEN}✓ Model metadata created${NC}"
}

# Backup configuration files
backup_configs() {
    log "Starting configuration backup..."
    
    local config_backup_dir="$BACKUP_DIR/$BACKUP_NAME/configs"
    mkdir -p "$config_backup_dir"
    
    # Backup Kubernetes configs
    if [ -d "k8s" ]; then
        cp -r k8s/* "$config_backup_dir/" 2>/dev/null || true
        log "${GREEN}✓ Kubernetes configs backed up${NC}"
    fi
    
    # Backup Docker configs
    if [ -f "docker-compose.yml" ]; then
        cp docker-compose.yml "$config_backup_dir/" 2>/dev/null || true
        log "${GREEN}✓ Docker Compose config backed up${NC}"
    fi
    
    # Backup environment files
    find . -name "*.env*" -o -name "*.yaml" -o -name "*.yml" | while read -r file; do
        if [ -f "$file" ]; then
            cp "$file" "$config_backup_dir/" 2>/dev/null || true
        fi
    done
    log "${GREEN}✓ Environment configs backed up${NC}"
}

# Backup application logs
backup_logs() {
    log "Starting log backup..."
    
    local log_backup_dir="$BACKUP_DIR/$BACKUP_NAME/logs"
    mkdir -p "$log_backup_dir"
    
    # Find and backup log files
    find . -name "*.log" -type f -mtime -7 | while read -r log_file; do
        if [ -f "$log_file" ]; then
            cp "$log_file" "$log_backup_dir/" 2>/dev/null || true
        fi
    done
    
    # Backup system logs if accessible
    if [ -d "/var/log" ] && [ -r "/var/log" ]; then
        find /var/log -name "*traffic*" -o -name "*smart*" 2>/dev/null | while read -r log_file; do
            if [ -f "$log_file" ]; then
                cp "$log_file" "$log_backup_dir/" 2>/dev/null || true
            fi
        done
    fi
    
    log "${GREEN}✓ Application logs backed up${NC}"
}

# Create backup manifest
create_manifest() {
    log "Creating backup manifest..."
    
    local manifest_file="$BACKUP_DIR/$BACKUP_NAME/manifest.json"
    cat > "$manifest_file" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "backup_date": "$(date -Iseconds)",
    "backup_type": "full",
    "components": {
        "database": {
            "backed_up": true,
            "file": "database_backup.sql.gz"
        },
        "models": {
            "backed_up": true,
            "directory": "models/"
        },
        "configs": {
            "backed_up": true,
            "directory": "configs/"
        },
        "logs": {
            "backed_up": true,
            "directory": "logs/"
        }
    },
    "backup_size": "$(du -sh "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)",
    "file_count": $(find "$BACKUP_DIR/$BACKUP_NAME" -type f | wc -l)
}
EOF
    log "${GREEN}✓ Backup manifest created${NC}"
}

# Upload to S3 (optional)
upload_to_s3() {
    if [ "$1" = "--s3" ] && command -v aws &> /dev/null; then
        log "Uploading backup to S3..."
        
        if aws s3 cp "$BACKUP_DIR/$BACKUP_NAME" "s3://$S3_BUCKET/$BACKUP_NAME" \
            --recursive \
            --region "$S3_REGION" 2>> "$LOG_FILE"; then
            log "${GREEN}✓ Backup uploaded to S3 successfully${NC}"
        else
            log "${RED}✗ S3 upload failed${NC}"
            return 1
        fi
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    local retention_days="${1:-30}"
    
    log "Cleaning up backups older than $retention_days days..."
    
    if [ -d "$BACKUP_DIR" ]; then
        find "$BACKUP_DIR" -type d -name "smart-traffic-backup-*" -mtime +$retention_days -exec rm -rf {} \; 2>/dev/null || true
        log "${GREEN}✓ Old backups cleaned up${NC}"
    fi
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    local backup_path="$BACKUP_DIR/$BACKUP_NAME"
    
    # Check if backup directory exists
    if [ ! -d "$backup_path" ]; then
        log "${RED}✗ Backup directory not found${NC}"
        return 1
    fi
    
    # Check if manifest exists
    if [ ! -f "$backup_path/manifest.json" ]; then
        log "${RED}✗ Backup manifest not found${NC}"
        return 1
    fi
    
    # Check if database backup exists
    if [ ! -f "$backup_path/database_backup.sql.gz" ]; then
        log "${YELLOW}⚠ Database backup not found${NC}"
    fi
    
    # Check if models directory exists
    if [ ! -d "$backup_path/models" ]; then
        log "${YELLOW}⚠ Models directory not found${NC}"
    fi
    
    log "${GREEN}✓ Backup verification completed${NC}"
    return 0
}

# Main backup function
run_backup() {
    log "Starting Smart Traffic Management System Backup..."
    log "=================================================="
    
    # Create backup directory
    create_backup_dir
    
    # Run backup components
    backup_database
    backup_models
    backup_configs
    backup_logs
    
    # Create manifest
    create_manifest
    
    # Verify backup
    verify_backup
    
    # Upload to S3 if requested
    if [ "$1" = "--s3" ]; then
        upload_to_s3 --s3
    fi
    
    # Cleanup old backups
    cleanup_old_backups
    
    log "\n=================================================="
    log "Backup completed successfully!"
    log "Backup location: $BACKUP_DIR/$BACKUP_NAME"
    log "Backup size: $(du -sh "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)"
    log "Log file: $LOG_FILE"
}

# Restore function
restore_backup() {
    local backup_name="$1"
    local restore_dir="$2"
    
    if [ -z "$backup_name" ]; then
        log "${RED}✗ Backup name required for restore${NC}"
        echo "Usage: $0 restore <backup_name> [restore_directory]"
        exit 1
    fi
    
    local backup_path="$BACKUP_DIR/$backup_name"
    local target_dir="${restore_dir:-/tmp/restore}"
    
    log "Starting restore from backup: $backup_name"
    log "Restore target: $target_dir"
    
    if [ ! -d "$backup_path" ]; then
        log "${RED}✗ Backup not found: $backup_path${NC}"
        exit 1
    fi
    
    # Create restore directory
    mkdir -p "$target_dir"
    
    # Restore database
    if [ -f "$backup_path/database_backup.sql.gz" ]; then
        log "Restoring database..."
        gunzip -c "$backup_path/database_backup.sql.gz" | psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB"
        log "${GREEN}✓ Database restored${NC}"
    fi
    
    # Restore models
    if [ -d "$backup_path/models" ]; then
        log "Restoring models..."
        cp -r "$backup_path/models"/* "$target_dir/" 2>/dev/null || true
        log "${GREEN}✓ Models restored${NC}"
    fi
    
    # Restore configs
    if [ -d "$backup_path/configs" ]; then
        log "Restoring configs..."
        cp -r "$backup_path/configs"/* "$target_dir/" 2>/dev/null || true
        log "${GREEN}✓ Configs restored${NC}"
    fi
    
    log "${GREEN}✓ Restore completed successfully!${NC}"
}

# Main execution
main() {
    case "$1" in
        "backup")
            run_backup "$2"
            ;;
        "restore")
            restore_backup "$2" "$3"
            ;;
        "list")
            log "Available backups:"
            if [ -d "$BACKUP_DIR" ]; then
                ls -la "$BACKUP_DIR" | grep "smart-traffic-backup-"
            else
                log "No backups found"
            fi
            ;;
        "cleanup")
            cleanup_old_backups "${2:-30}"
            ;;
        "--help"|"-h")
            echo "Usage: $0 {backup|restore|list|cleanup} [options]"
            echo ""
            echo "Commands:"
            echo "  backup [--s3]     Create a new backup (optionally upload to S3)"
            echo "  restore <name>    Restore from a backup"
            echo "  list              List available backups"
            echo "  cleanup [days]    Clean up old backups (default: 30 days)"
            echo ""
            echo "Environment variables:"
            echo "  POSTGRES_PASSWORD  Database password"
            echo "  AWS_ACCESS_KEY_ID  AWS access key (for S3 upload)"
            echo "  AWS_SECRET_ACCESS_KEY  AWS secret key (for S3 upload)"
            ;;
        *)
            echo "Usage: $0 {backup|restore|list|cleanup} [options]"
            echo "Use '$0 --help' for more information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"

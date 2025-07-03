#!/usr/bin/env python3
"""Quick database access commands."""

import sqlite3
import json
from datetime import datetime

def show_all_projects():
    """Show all projects."""
    conn = sqlite3.connect('test_api_keys.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, created_at, total_requests, total_characters 
        FROM projects 
        ORDER BY created_at DESC
    """)
    
    projects = cursor.fetchall()
    
    print("📁 All Projects:")
    for proj in projects:
        print(f"  ID: {proj[0]}")
        print(f"  Name: {proj[1]}")
        print(f"  Created: {proj[2]}")
        print(f"  Requests: {proj[3] or 0}")
        print(f"  Characters: {proj[4] or 0}")
        print("-" * 40)
    
    conn.close()

def show_all_api_keys():
    """Show all API keys with project names."""
    conn = sqlite3.connect('test_api_keys.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ak.id, p.name as project_name, ak.key_prefix, 
               ak.rate_limit_tier, ak.is_active, ak.created_at,
               ak.usage_count, ak.total_characters, ak.last_used_at
        FROM api_keys ak
        JOIN projects p ON ak.project_id = p.id
        ORDER BY ak.created_at DESC
    """)
    
    keys = cursor.fetchall()
    
    print("🔑 All API Keys:")
    for key in keys:
        print(f"  Key ID: {key[0]}")
        print(f"  Project: {key[1]}")
        print(f"  Key Prefix: {key[2]}")
        print(f"  Tier: {key[3]}")
        print(f"  Active: {key[4]}")
        print(f"  Created: {key[5]}")
        print(f"  Usage: {key[6] or 0}")
        print(f"  Characters: {key[7] or 0}")
        print(f"  Last Used: {key[8] or 'Never'}")
        print("-" * 40)
    
    conn.close()

def search_projects(search_term):
    """Search projects by name."""
    conn = sqlite3.connect('test_api_keys.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, created_at 
        FROM projects 
        WHERE name LIKE ?
        ORDER BY created_at DESC
    """, (f'%{search_term}%',))
    
    projects = cursor.fetchall()
    
    print(f"🔍 Projects matching '{search_term}':")
    for proj in projects:
        print(f"  {proj[1]} (ID: {proj[0]}, Created: {proj[2]})")
    
    conn.close()

def get_usage_stats():
    """Get usage statistics."""
    conn = sqlite3.connect('test_api_keys.db')
    cursor = conn.cursor()
    
    # Overall stats
    cursor.execute("SELECT COUNT(*) FROM projects")
    total_projects = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM api_keys WHERE is_active = 1")
    active_keys = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(usage_count) FROM api_keys")
    total_requests = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT SUM(total_characters) FROM api_keys")
    total_chars = cursor.fetchone()[0] or 0
    
    print("📊 Usage Statistics:")
    print(f"  Total Projects: {total_projects}")
    print(f"  Active API Keys: {active_keys}")
    print(f"  Total Requests: {total_requests}")
    print(f"  Total Characters: {total_chars}")
    
    # Top projects by usage
    cursor.execute("""
        SELECT p.name, SUM(ak.usage_count) as requests, SUM(ak.total_characters) as chars
        FROM projects p
        LEFT JOIN api_keys ak ON p.id = ak.project_id
        GROUP BY p.id, p.name
        ORDER BY requests DESC
        LIMIT 5
    """)
    
    top_projects = cursor.fetchall()
    
    print("\n🏆 Top Projects by Usage:")
    for proj in top_projects:
        print(f"  {proj[0]}: {proj[1] or 0} requests, {proj[2] or 0} chars")
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python quick_db_access.py projects       # Show all projects")
        print("  python quick_db_access.py keys           # Show all API keys") 
        print("  python quick_db_access.py search <term>  # Search projects")
        print("  python quick_db_access.py stats          # Usage statistics")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "projects":
        show_all_projects()
    elif command == "keys":
        show_all_api_keys()
    elif command == "search" and len(sys.argv) > 2:
        search_projects(sys.argv[2])
    elif command == "stats":
        get_usage_stats()
    else:
        print("Unknown command. Use: projects, keys, search <term>, or stats")
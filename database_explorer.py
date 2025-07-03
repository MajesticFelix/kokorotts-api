#!/usr/bin/env python3
"""Database explorer script for KokoroTTS API."""

import sqlite3
import json
from datetime import datetime
import os

def explore_database():
    """Explore the KokoroTTS database."""
    
    db_path = "./test_api_keys.db"
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return
    
    print(f"🗄️ Exploring database: {db_path}")
    print("=" * 50)
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        # Get table information
        print("📋 Database Tables:")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table['name']
            print(f"  - {table_name}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            print(f"    Columns:")
            for col in columns:
                print(f"      {col['name']} ({col['type']}) {'NOT NULL' if col['notnull'] else 'NULL'}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name};")
            count = cursor.fetchone()['count']
            print(f"    Rows: {count}")
            print()
        
        # Show projects data
        print("📁 Projects Data:")
        cursor.execute("SELECT * FROM projects;")
        projects = cursor.fetchall()
        
        if projects:
            for project in projects:
                print(f"  Project ID: {project['id']}")
                print(f"  Name: {project['name']}")
                print(f"  Created: {project['created_at']}")
                print(f"  Usage Count: {project['usage_count']}")
                print(f"  Total Characters: {project['total_characters']}")
                print(f"  Last Used: {project['last_used_at']}")
                print("-" * 30)
        else:
            print("  No projects found")
        
        print()
        
        # Show API keys data
        print("🔑 API Keys Data:")
        cursor.execute("SELECT * FROM api_keys;")
        api_keys = cursor.fetchall()
        
        if api_keys:
            for key in api_keys:
                print(f"  Key ID: {key['id']}")
                print(f"  Project ID: {key['project_id']}")
                print(f"  Key Hash: {key['key_hash'][:20]}...")
                print(f"  Rate Limit Tier: {key['rate_limit_tier']}")
                print(f"  Active: {key['is_active']}")
                print(f"  Created: {key['created_at']}")
                print(f"  Usage Count: {key['usage_count']}")
                print(f"  Total Characters: {key['total_characters']}")
                print(f"  Last Used: {key['last_used_at']}")
                print("-" * 30)
        else:
            print("  No API keys found")
        
        # Show recent activity
        print()
        print("⏰ Recent Activity (Last 10 API keys by creation):")
        cursor.execute("""
            SELECT ak.id, p.name as project_name, ak.rate_limit_tier, 
                   ak.created_at, ak.usage_count, ak.total_characters
            FROM api_keys ak
            JOIN projects p ON ak.project_id = p.id
            ORDER BY ak.created_at DESC
            LIMIT 10;
        """)
        
        recent = cursor.fetchall()
        if recent:
            for item in recent:
                print(f"  {item['project_name']} | {item['rate_limit_tier']} | "
                      f"Usage: {item['usage_count']} | Chars: {item['total_characters']} | "
                      f"Created: {item['created_at']}")
        else:
            print("  No recent activity")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error exploring database: {e}")

def run_custom_query():
    """Run a custom SQL query."""
    
    db_path = "./test_api_keys.db"
    
    print("🔍 Custom Query Mode")
    print("Enter SQL query (or 'quit' to exit):")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    while True:
        query = input("SQL> ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
            
        try:
            cursor.execute(query)
            
            if query.upper().startswith('SELECT'):
                results = cursor.fetchall()
                if results:
                    # Print column headers
                    columns = [description[0] for description in cursor.description]
                    print(" | ".join(columns))
                    print("-" * (len(" | ".join(columns))))
                    
                    # Print data
                    for row in results:
                        print(" | ".join(str(row[col]) for col in columns))
                else:
                    print("No results found")
            else:
                conn.commit()
                print(f"Query executed successfully. Rows affected: {cursor.rowcount}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        run_custom_query()
    else:
        explore_database()
        
        print("\n" + "=" * 50)
        print("💡 Usage:")
        print("  python database_explorer.py        # Explore database")
        print("  python database_explorer.py query  # Custom queries")
        print("\n📖 Example queries:")
        print("  SELECT * FROM projects WHERE name LIKE '%test%';")
        print("  SELECT COUNT(*) FROM api_keys WHERE is_active = 1;")
        print("  SELECT p.name, COUNT(ak.id) as key_count FROM projects p")
        print("    LEFT JOIN api_keys ak ON p.id = ak.project_id GROUP BY p.id;")
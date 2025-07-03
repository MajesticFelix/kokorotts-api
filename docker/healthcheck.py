#!/usr/bin/env python3
"""
Health check script for KokoroTTS API Docker container.
This script is used by Docker's HEALTHCHECK instruction to verify the container is running properly.
"""

import os
import sys
import time
import json
import socket
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


def get_config():
    """Get configuration from environment variables."""
    return {
        'host': os.getenv('HOST', '0.0.0.0'),
        'port': int(os.getenv('PORT', 8000)),
        'timeout': int(os.getenv('HEALTH_CHECK_TIMEOUT', 10)),
        'max_retries': int(os.getenv('HEALTH_CHECK_RETRIES', 3)),
        'retry_delay': float(os.getenv('HEALTH_CHECK_RETRY_DELAY', 1.0)),
    }


def check_port_open(host, port, timeout=5):
    """Check if the application port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, socket.error, OSError):
        return False


def check_health_endpoint(host, port, timeout=10):
    """Check the /health endpoint."""
    url = f"http://{host}:{port}/health"
    
    try:
        request = Request(url, headers={'User-Agent': 'Docker-Health-Check/1.0'})
        with urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                return data.get('status') == 'healthy'
            return False
    except (URLError, HTTPError, json.JSONDecodeError, KeyError):
        return False


def check_basic_endpoint(host, port, timeout=10):
    """Check the root endpoint as fallback."""
    url = f"http://{host}:{port}/"
    
    try:
        request = Request(url, headers={'User-Agent': 'Docker-Health-Check/1.0'})
        with urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (URLError, HTTPError):
        return False


def check_tts_functionality(host, port, timeout=15):
    """Basic check of TTS functionality by testing the voices endpoint."""
    url = f"http://{host}:{port}/v1/audio/voices"
    
    try:
        request = Request(url, headers={'User-Agent': 'Docker-Health-Check/1.0'})
        with urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                # Check if we have voices available
                return 'voices' in data and len(data['voices']) > 0
            return False
    except (URLError, HTTPError, json.JSONDecodeError, KeyError):
        return False


def perform_health_check():
    """Perform comprehensive health check."""
    config = get_config()
    host = config['host']
    port = config['port']
    timeout = config['timeout']
    max_retries = config['max_retries']
    retry_delay = config['retry_delay']
    
    # If host is 0.0.0.0, use localhost for health check
    check_host = 'localhost' if host == '0.0.0.0' else host
    
    print(f"Health check starting for {check_host}:{port}")
    
    for attempt in range(max_retries):
        try:
            # Step 1: Check if port is open
            if not check_port_open(check_host, port, timeout=5):
                print(f"Attempt {attempt + 1}: Port {port} is not open")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return False
            
            # Step 2: Check health endpoint
            if check_health_endpoint(check_host, port, timeout):
                print("Health endpoint check passed")
                
                # Step 3: Check TTS functionality (optional, for deeper health check)
                if check_tts_functionality(check_host, port, timeout):
                    print("TTS functionality check passed")
                    return True
                else:
                    print("TTS functionality check failed, but basic health is OK")
                    return True  # Still consider healthy if basic endpoints work
            
            # Fallback: Check basic endpoint
            if check_basic_endpoint(check_host, port, timeout):
                print("Basic endpoint check passed (health endpoint failed)")
                return True
            
            print(f"Attempt {attempt + 1}: All checks failed")
            
        except Exception as e:
            print(f"Attempt {attempt + 1}: Health check error: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    print("All health check attempts failed")
    return False


def main():
    """Main health check function."""
    start_time = time.time()
    
    try:
        is_healthy = perform_health_check()
        duration = time.time() - start_time
        
        if is_healthy:
            print(f"✅ Health check passed in {duration:.2f}s")
            sys.exit(0)
        else:
            print(f"❌ Health check failed after {duration:.2f}s")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("Health check interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Health check error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
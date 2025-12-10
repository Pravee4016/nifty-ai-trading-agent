
import os

def sync_env():
    env_vars = {}
    print("Reading .env...")
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    # Handle valus with = sign
                    parts = line.strip().split('=', 1)
                    if len(parts) == 2:
                        key, value = parts
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        env_vars[key] = value
    except FileNotFoundError:
        print('ERROR: .env not found')
        return

    print("Writing .env.yaml...")
    try:
        with open('.env.yaml', 'w') as f:
            for k, v in env_vars.items():
                # Escape quotes in value if needed
                v_clean = v.replace('"', '\\"')
                f.write(f'{k}: "{v_clean}"\n')
        print('✅ Synced .env to .env.yaml')
    except Exception as e:
        print(f'ERROR writing .env.yaml: {e}')

if __name__ == "__main__":
    sync_env()

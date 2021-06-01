from unityagents import UnityEnvironment


def create_env_wrapper(config) -> UnityEnvironment:
    env_name = config['env']
    if env_name == "Reacher":
        return UnityEnvironment(file_name=config['unity_env_path'])

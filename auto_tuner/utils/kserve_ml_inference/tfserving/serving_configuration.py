def get_serving_configuration(model_name: str, base_path: str, model_platform: str, model_version: int):
    return f"""
    model_config_list {{
      config {{
        name: '{model_name}'
        base_path: '{base_path}'
        model_platform: '{model_platform}'
        model_version_policy {{
          specific {{
            versions: {model_version}
          }}
        }}
      }}
    }}
    """

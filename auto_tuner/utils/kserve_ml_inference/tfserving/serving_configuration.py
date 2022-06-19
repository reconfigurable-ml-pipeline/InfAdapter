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


def get_batch_configuration(max_batch_size: int, num_batch_threads: int):
    return f"""
    max_batch_size {{ value: {max_batch_size} }}
    batch_timeout_micros {{ value: 0 }}
    max_enqueued_batches {{ value: 1000000 }}
    num_batch_threads {{ value: {num_batch_threads} }}
    """
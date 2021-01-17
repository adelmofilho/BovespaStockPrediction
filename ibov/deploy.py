import boto3


def get_deploy_config(config):
    
    # Get configs
    role = config.get("deploy").get("role")
    codename = config.get("deploy").get("codename")
    instance_type = config.get("deploy").get("instance_type")
    instance_count = config.get("deploy").get("instance_count")
    training_job_name = config.get("deploy").get("training_job_name")
    
    # Get info about training job
    client  = boto3.client("sagemaker")
    training_job = client.describe_training_job(TrainingJobName=training_job_name)
    
    image = training_job["AlgorithmSpecification"]["TrainingImage"]
    model_data = training_job["ModelArtifacts"]["S3ModelArtifacts"]
    source_data = training_job["HyperParameters"]["sagemaker_submit_directory"].replace("\"", "")
    
    # Set deploy parameters dictionaries
    
    primary_container = {
        'ContainerHostname': codename,
        'Image': image,
        'ImageConfig': {
            'RepositoryAccessMode': 'Platform'
        },
        'Mode': 'SingleModel',
        'ModelDataUrl': model_data,
        'Environment': {
              "SAGEMAKER_PROGRAM": "train.py",
              'SAGEMAKER_SUBMIT_DIRECTORY': source_data
          },
    }
    
    primary_variant = {
        'VariantName': codename,
        'ModelName': codename,
        'InitialInstanceCount': instance_count,
        'InstanceType': instance_type
    }
    
    response = {
        "codename": codename,
        "role": role,
        "primary_container": primary_container,
        "primary_variant": primary_variant
    }
    
    return response


def build_endpoint(deploy_config):
    
    codename = deploy_config.get("codename")
    
    client  = boto3.client("sagemaker")    
    
    model = client.create_model(ModelName=codename,
                                PrimaryContainer=deploy_config.get("primary_container"),
                                ExecutionRoleArn=deploy_config.get("role"))
    
    config = client.create_endpoint_config(EndpointConfigName=codename,
                                           ProductionVariants=[deploy_config.get("primary_variant")])
                                           
    endpoint = client.create_endpoint(EndpointName=codename,
                                      EndpointConfigName=codename)
    
    return model, config, endpoint


def kill_endpoint(config):
    
    codename = config.get("deploy").get("codename")
    
    client = boto3.client("sagemaker")    
    model = client.delete_model(ModelName=codename)
    config = client.delete_endpoint_config(EndpointConfigName=codename)
    endpoint = client.delete_endpoint(EndpointName=codename)
    
    return model, config, endpoint
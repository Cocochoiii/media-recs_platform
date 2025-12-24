#!/usr/bin/env python3
"""
AWS SageMaker Deployment Script

Deploy recommendation models to SageMaker for production inference.
"""

import os
import sys
import argparse
import logging
import json
import tarfile
import tempfile
from datetime import datetime
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerDeployer:
    """Deploy models to AWS SageMaker."""
    
    def __init__(
        self,
        region: str = "us-east-1",
        role_arn: Optional[str] = None
    ):
        self.region = region
        self.role_arn = role_arn or os.environ.get("SAGEMAKER_ROLE_ARN")
        
        if not self.role_arn:
            raise ValueError("SageMaker role ARN not provided")
        
        self.session = boto3.Session(region_name=region)
        self.sagemaker = self.session.client("sagemaker")
        self.s3 = self.session.client("s3")
    
    def package_model(
        self,
        model_dir: str,
        output_path: str
    ) -> str:
        """
        Package model artifacts into tar.gz for SageMaker.
        
        Args:
            model_dir: Directory containing model files
            output_path: Output path for tar.gz
            
        Returns:
            Path to created archive
        """
        logger.info(f"Packaging model from {model_dir}")
        
        with tarfile.open(output_path, "w:gz") as tar:
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                tar.add(file_path, arcname=file)
        
        logger.info(f"Created model package: {output_path}")
        return output_path
    
    def upload_to_s3(
        self,
        local_path: str,
        bucket: str,
        key: str
    ) -> str:
        """Upload file to S3."""
        logger.info(f"Uploading to s3://{bucket}/{key}")
        
        self.s3.upload_file(local_path, bucket, key)
        
        return f"s3://{bucket}/{key}"
    
    def create_model(
        self,
        model_name: str,
        model_data_url: str,
        image_uri: str,
        environment: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create SageMaker model.
        
        Args:
            model_name: Name for the model
            model_data_url: S3 URL to model artifacts
            image_uri: Docker image URI for inference
            environment: Environment variables
            
        Returns:
            Model ARN
        """
        logger.info(f"Creating SageMaker model: {model_name}")
        
        create_model_params = {
            "ModelName": model_name,
            "ExecutionRoleArn": self.role_arn,
            "PrimaryContainer": {
                "Image": image_uri,
                "ModelDataUrl": model_data_url,
            }
        }
        
        if environment:
            create_model_params["PrimaryContainer"]["Environment"] = environment
        
        try:
            response = self.sagemaker.create_model(**create_model_params)
            logger.info(f"Created model: {response['ModelArn']}")
            return response["ModelArn"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                # Model might already exist, delete and recreate
                logger.warning(f"Model {model_name} exists, recreating...")
                self.sagemaker.delete_model(ModelName=model_name)
                response = self.sagemaker.create_model(**create_model_params)
                return response["ModelArn"]
            raise
    
    def create_endpoint_config(
        self,
        config_name: str,
        model_name: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1
    ) -> str:
        """Create endpoint configuration."""
        logger.info(f"Creating endpoint config: {config_name}")
        
        try:
            response = self.sagemaker.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        "VariantName": "primary",
                        "ModelName": model_name,
                        "InstanceType": instance_type,
                        "InitialInstanceCount": instance_count,
                        "InitialVariantWeight": 1.0,
                    }
                ]
            )
            return response["EndpointConfigArn"]
        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                logger.warning(f"Config {config_name} exists, recreating...")
                self.sagemaker.delete_endpoint_config(
                    EndpointConfigName=config_name
                )
                response = self.sagemaker.create_endpoint_config(
                    EndpointConfigName=config_name,
                    ProductionVariants=[
                        {
                            "VariantName": "primary",
                            "ModelName": model_name,
                            "InstanceType": instance_type,
                            "InitialInstanceCount": instance_count,
                            "InitialVariantWeight": 1.0,
                        }
                    ]
                )
                return response["EndpointConfigArn"]
            raise
    
    def create_endpoint(
        self,
        endpoint_name: str,
        config_name: str,
        wait: bool = True
    ) -> str:
        """Create or update endpoint."""
        logger.info(f"Creating endpoint: {endpoint_name}")
        
        try:
            # Check if endpoint exists
            self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            
            # Update existing endpoint
            logger.info("Endpoint exists, updating...")
            self.sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        except ClientError:
            # Create new endpoint
            self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        
        if wait:
            logger.info("Waiting for endpoint to be ready...")
            waiter = self.sagemaker.get_waiter("endpoint_in_service")
            waiter.wait(
                EndpointName=endpoint_name,
                WaiterConfig={"Delay": 30, "MaxAttempts": 60}
            )
        
        response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint ready: {response['EndpointArn']}")
        return response["EndpointArn"]
    
    def deploy(
        self,
        model_dir: str,
        s3_bucket: str,
        endpoint_name: str,
        image_uri: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1
    ) -> Dict[str, str]:
        """
        Full deployment pipeline.
        
        Args:
            model_dir: Local directory with model artifacts
            s3_bucket: S3 bucket for model storage
            endpoint_name: Name for the endpoint
            image_uri: Docker image for inference
            instance_type: EC2 instance type
            instance_count: Number of instances
            
        Returns:
            Dictionary with deployment info
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"{endpoint_name}-model-{timestamp}"
        config_name = f"{endpoint_name}-config-{timestamp}"
        
        # Package and upload model
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
            package_path = f.name
        
        try:
            self.package_model(model_dir, package_path)
            
            s3_key = f"models/{endpoint_name}/{timestamp}/model.tar.gz"
            model_data_url = self.upload_to_s3(package_path, s3_bucket, s3_key)
            
            # Create model
            model_arn = self.create_model(model_name, model_data_url, image_uri)
            
            # Create endpoint config
            config_arn = self.create_endpoint_config(
                config_name, model_name, instance_type, instance_count
            )
            
            # Create/update endpoint
            endpoint_arn = self.create_endpoint(endpoint_name, config_name)
            
            return {
                "model_name": model_name,
                "model_arn": model_arn,
                "config_name": config_name,
                "config_arn": config_arn,
                "endpoint_name": endpoint_name,
                "endpoint_arn": endpoint_arn,
                "model_data_url": model_data_url
            }
        finally:
            if os.path.exists(package_path):
                os.remove(package_path)
    
    def delete_endpoint(self, endpoint_name: str):
        """Delete endpoint and associated resources."""
        logger.info(f"Deleting endpoint: {endpoint_name}")
        
        try:
            # Get endpoint config
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            config_name = response["EndpointConfigName"]
            
            # Delete endpoint
            self.sagemaker.delete_endpoint(EndpointName=endpoint_name)
            
            # Wait for deletion
            waiter = self.sagemaker.get_waiter("endpoint_deleted")
            waiter.wait(EndpointName=endpoint_name)
            
            # Delete config
            self.sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            
            logger.info("Endpoint deleted successfully")
        except ClientError as e:
            logger.error(f"Error deleting endpoint: {e}")
            raise


class SageMakerInference:
    """Client for SageMaker inference."""
    
    def __init__(self, endpoint_name: str, region: str = "us-east-1"):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.Session(region_name=region).client(
            "sagemaker-runtime"
        )
    
    def predict(self, data: Dict) -> Dict:
        """Make prediction request."""
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(data)
        )
        
        result = json.loads(response["Body"].read().decode())
        return result
    
    def batch_predict(self, data_list: list, batch_size: int = 100) -> list:
        """Make batch predictions."""
        results = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps({"instances": batch})
            )
            batch_results = json.loads(response["Body"].read().decode())
            results.extend(batch_results.get("predictions", []))
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Deploy to SageMaker")
    parser.add_argument("--action", choices=["deploy", "delete"], required=True)
    parser.add_argument("--model-dir", type=str, help="Local model directory")
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket")
    parser.add_argument("--endpoint-name", type=str, default="media-recommender")
    parser.add_argument("--image-uri", type=str, help="Docker image URI")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--role-arn", type=str)
    
    args = parser.parse_args()
    
    deployer = SageMakerDeployer(region=args.region, role_arn=args.role_arn)
    
    if args.action == "deploy":
        if not all([args.model_dir, args.s3_bucket, args.image_uri]):
            parser.error("deploy requires --model-dir, --s3-bucket, and --image-uri")
        
        result = deployer.deploy(
            model_dir=args.model_dir,
            s3_bucket=args.s3_bucket,
            endpoint_name=args.endpoint_name,
            image_uri=args.image_uri,
            instance_type=args.instance_type,
            instance_count=args.instance_count
        )
        
        print("\nDeployment successful!")
        print(json.dumps(result, indent=2))
    
    elif args.action == "delete":
        deployer.delete_endpoint(args.endpoint_name)
        print("Endpoint deleted successfully")


if __name__ == "__main__":
    main()

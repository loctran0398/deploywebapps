import boto3
from botocore.exceptions import ClientError


class S3Load(object):
    def __init__(self):
        pass

    @staticmethod
    def download(bucket_name, filename, local):
        client = boto3.client('s3')
        try:
            client.download_file(bucket_name, filename, local)
        except ClientError as e:
            raise e

    @staticmethod
    def read_file(path):
        client = boto3.client('s3')
        obj = {}
        try:
            bucket_name = path.replace('s3://', '').split('/')[0]
            key = "/".join(path.replace('s3://', '').split('/')[1:])
            obj = client.get_object(Bucket=bucket_name, Key=key)
        except Exception as e:
            raise e
        return obj.get('Body', None)

    @staticmethod
    def upload_file(local_path, s3_path):
        client = boto3.client('s3')
        try:
            bucket_name = s3_path.replace('s3://', '').split('/')[0]
            key = "/".join(s3_path.replace('s3://', '').split('/')[1:])
            obj = client.upload_file(local_path, bucket_name, key)
        except Exception as e:
            raise e

    @staticmethod
    def upload_text_data(data, s3_path):
        """
        Function upload data from variable such as json string or csv string
        Args:
          data (str): json string object or csv string
          s3_path (str): path to S3 including the filename
        Returns:
        """
        try:
            cli = boto3.client('s3')
            bucket_name = s3_path.replace('s3://', '').split('/')[0]
            bucket_key = "/".join(s3_path.replace('s3://%s/' % bucket_name, '')
                                  .split('/')[:-1])
            filename = s3_path.replace('s3://', '').split('/')[-1]
            print(
                "Uploading: %s to S3 with bucket name: %s and bucket key: %s..." % (
                    filename, bucket_name, bucket_key))
            cli.put_object(Bucket=bucket_name,
                           Key="%s/%s" % (bucket_key, filename),
                           Body=data)
        except Exception as e:
            print(e)

    @staticmethod
    def list_all_file_objects(path_to_directory):
        """
        Function get all information file objects from S3 directory
        Args:
          path_to_directory (str): path to S3 directory
        Returns:
          file_objects list[S3 Bucket Object]: list of file object from Bucket S3
        """
        file_objects = []
        try:
            s3resource = boto3.resource('s3')
            bucket_name = path_to_directory.replace('s3://', '').split('/')[0]
            bucket = s3resource.Bucket(bucket_name)
            prefix = path_to_directory.split(bucket_name + "/")
            file_objects = bucket.objects.filter(Prefix=prefix[-1])
        except Exception as e:
            print(e)
        return file_objects

    @staticmethod
    def list_all_json_files(path_to_directory):
        file_objects = S3Load.list_all_file_objects(path_to_directory)
        bucket_name = path_to_directory.replace('s3://', '').split('/')[0]

        json_files = []
        for obj in file_objects:
            prefix_filename = obj.key
            if prefix_filename.lower().endswith('.json'):
                path_to_file = 's3://%s/%s' % (bucket_name, prefix_filename)
                json_files.append(path_to_file)

        return json_files

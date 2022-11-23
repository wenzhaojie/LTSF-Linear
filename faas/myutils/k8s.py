import subprocess


class Kubernetes:
    def __init__(self):
        pass


    def get_replicas(self, deployment, namespace="openfaas-fn"):
        cmd = f"kubectl get deployment -n {namespace} | grep {deployment}"
        output = subprocess.getoutput(cmd)
        result = []
        for item in output.split(" "):
            if item != '':
                result.append(item)

        print(result)
        replicas = result[1].split('/')[0]



if __name__ == "__main__":
    client = Kubernetes()
    client.get_replicas(deployment="hello-python")
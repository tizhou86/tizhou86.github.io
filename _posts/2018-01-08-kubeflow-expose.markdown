#Kubeflow Expose

##Kubeflow introduction

The Kubeflow project is dedicated to making Machine Learning easy to set up with Kubernetes, portable and scalable. The goal is not to recreate other services, but to provide a straightforward way for spinning up best of breed OSS solutions. Kubernetes is an open-source platform for automating deployment, scaling, and management of containerised applications.

Because Kubeflow relies on Kubernetes, it runs wherever Kubernetes runs such as bare-metal servers, or cloud providers such as Google. Details of the project can be found at https://github.com/google/kubeflow



##Kubeflow Components

Kubeflow has three core components.

* TF Job Operator and Controller: Extension to Kubernetes to simplify deployment of distributed TensorFlow workloads. By using an Operator, Kubeflow is capable of automatically configuring the master, worker and parameterized server configuration. Workloads can be deployed with a TFJob.
* TF Hub: Running instances of JupyterHub, enabling you to work with Jupyter Notebooks.
* Model Server: Deploying a trained TensorFlow models for clients to access and use for future predictions.



##Kubeflow Deployment

####Codebase

```
master $ ls -lha kubeflow/components/
total 20K
drwxr-xr-x 5 root root 4.0K Jan  8 07:05 .
drwxr-xr-x 5 root root 4.0K Jan  8 07:05 ..
drwxr-xr-x 4 root root 4.0K Jan  8 07:05 jupyterhub
drwxr-xr-x 4 root root 4.0K Jan  8 07:05 k8s-model-server
drwxr-xr-x 2 root root 4.0K Jan  8 07:05 tf-controller
```

####Deploy Components

```
master $ kubectl apply -f kubeflow/components/ -R
configmap "jupyterhub-config" created
service "tf-hub-0" created
statefulset "tf-hub" created
role "edit-pod" created
rolebinding "edit-pods" created
service "tf-hub-lb" created
deployment "model-server" created
service "model-service" created
configmap "tf-job-operator-config" created
serviceaccount "tf-job-operator" created
clusterrole "tf-job-operator" created
clusterrolebinding "tf-job-operator" created
deployment "tf-job-operator" created
```

####Show Components

```
master $ kubectl get all
NAME                     DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
deploy/model-server      3         3         3            3           5m
deploy/tf-job-operator   1         1         1            1           5m

NAME                            DESIRED   CURRENT   READY     AGE
rs/model-server-584cf76db9      3         3         3         5m
rs/tf-job-operator-6f7ccdfd4d   1         1         1         5m

NAME                     DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
deploy/model-server      3         3         3            3           5m
deploy/tf-job-operator   1         1         1            1           5m

NAME                  DESIRED   CURRENT   AGE
statefulsets/tf-hub   1         1         5m

NAME                            DESIRED   CURRENT   READY     AGE
rs/model-server-584cf76db9      3         3         3         5m
rs/tf-job-operator-6f7ccdfd4d   1         1         1         5m

NAME                                  READY     STATUS    RESTARTS   AGE
po/model-server-584cf76db9-lj57h      1/1       Running   0          5m
po/model-server-584cf76db9-r5c7h      1/1       Running   0          5m
po/model-server-584cf76db9-vgxm4      1/1       Running   0          5m
po/tf-hub-0                           1/1       Running   0          5m
po/tf-job-operator-6f7ccdfd4d-f97gp   1/1       Running   0          5m

NAME                TYPE           CLUSTER-IP     EXTERNAL-IP                 PORT(S)          AGE
svc/kubernetes      ClusterIP      10.96.0.1      <none>                      443/TCP          16m
svc/model-service   LoadBalancer   10.110.33.70   <pending>                   9000:31006/TCP   5m
svc/tf-hub-0        ClusterIP      None           <none>                      8000/TCP         5m
svc/tf-hub-lb       LoadBalancer   10.105.56.93   172.17.0.116,172.17.0.120   80:31005/TCP     5m
```

##Tensorflow Example

###Tensorflow Job

####Example Tensorflow Application

TensorFlow workload that performs a matrix multiplication across the defined workers and parameter servers.

https://github.com/tensorflow/k8s/tree/master/examples/tf_sample

```
for job_name in cluster_spec.keys():
  for i in range(len(cluster_spec[job_name])):
    d = "/job:{0}/task:{1}".format(job_name, i)
    with tf.device(d):
      a = tf.constant(range(width * height), shape=[height, width])
      b = tf.constant(range(width * height), shape=[height, width])
      c = tf.multiply(a, b)
      results.append(c)
```

####Deploy Tensorflow Job (TFJob)

TfJob provides a Kubeflow custom resource that makes it easy to run distributed or non-distributed TensorFlow jobs on Kubernetes. The TFJob controller takes a YAML specification for a master, parameter servers, and workers to help run distributed computation.

A Custom Resource Definition (CRD) provides the ability to create and manage TF Jobs in the same fashion as built-in Kubernetes resources. Once deployed, the CRD can configure the TensorFlow job, allowing users to focus on machine learning instead of infrastructure.

The definition defines three components:

* Master: Each job must have one master. The master will coordinate training operations execution between workers.

* Worker: A job can have 0 to N workers. Each worker process runs the same model, providing parameters for processing to a Parameter Server.

* PS: A job can have 0 to N parameter servers. Parameter server enables you to scale your model across multiple machines.

```
master $ cat example.yaml
apiVersion: "tensorflow.org/v1alpha1"
kind: "TfJob"
metadata:
  name: "example-job"
spec:
  replicaSpecs:
    - replicas: 1
      tfReplicaType: MASTER
      template:
        spec:
          containers:
            - image: gcr.io/tf-on-k8s-dogfood/tf_sample:dc944ff
              name: tensorflow
          restartPolicy: OnFailure
    - replicas: 1
      tfReplicaType: WORKER
      template:
        spec:
          containers:
            - image: gcr.io/tf-on-k8s-dogfood/tf_sample:dc944ff
              name: tensorflow
          restartPolicy: OnFailure
    - replicas: 2
      tfReplicaType: PS
```

Create the job

```
master $ kubectl apply -f example.yaml
tfjob "example-job" created
```

####View Job Progress and Results

Checkout the status of job.Once the TensorFlow job has been completed, the master is marked as successful. Keep running the job command to see when it finishes.

```
master $ kubectl get job
NAME                        DESIRED   SUCCESSFUL   AGE
example-job-master-l7yx-0   1         0            28s
example-job-ps-l7yx-0       1         0            26s
example-job-ps-l7yx-1       1         0            24s
example-job-worker-l7yx-0   1         0            26s
```

The master is responsible for coordinating the execution and aggregating the results. Under the covers, the completed workloads can be listed using 

```
master $ kubectl get pods -a | grep Completed
example-job-master-l7yx-0-kcc5l    0/1       Completed   0          1m
```

The results are outputted to STDOUT, viewable using kubectl logs.
The command below will output the results:


```
master $ kubectl logs $(kubectl get pods -a | grep Completed | tr -s ' ' | cut -d ' ' -f 1)
INFO:root:Tensorflow version: 1.3.0-rc2
INFO:root:Tensorflow git version: v1.3.0-rc1-27-g2784b1c
INFO:root:tf_config: {u'environment': u'cloud', u'cluster': {u'worker': [u'example-job-worker-l7yx-0:2222'], u'ps': [u'example-job-ps-l7yx-0:2222', u'example-job-ps-l7yx-1:2222'], u'master': [u'example-job-master-l7yx-0:2222']}, u'task': {u'index': 0, u'type': u'master'}}
INFO:root:task: {u'index': 0, u'type': u'master'}
INFO:root:cluster_spec: {u'worker': [u'example-job-worker-l7yx-0:2222'], u'ps': [u'example-job-ps-l7yx-0:2222', u'example-job-ps-l7yx-1:2222'], u'master': [u'example-job-master-l7yx-0:2222']}
INFO:root:server_def: cluster {
  job {
    name: "master"
    tasks {
      value: "example-job-master-l7yx-0:2222"
    }
  }
  job {
    name: "ps"
    tasks {
      value: "example-job-ps-l7yx-0:2222"
    }
    tasks {
      key: 1
      value: "example-job-ps-l7yx-1:2222"
    }
  }
  job {
    name: "worker"
    tasks {
      value: "example-job-worker-l7yx-0:2222"
    }
  }
}
job_name: "master"
protocol: "grpc"

INFO:root:Building server.
2018-01-08 07:47:35.955839: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-08 07:47:35.955956: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-08 07:47:35.955980: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-01-08 07:47:35.990985: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job master -> {0 -> localhost:2222}
2018-01-08 07:47:35.991056: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job ps -> {0 ->example-job-ps-l7yx-0:2222, 1 -> example-job-ps-l7yx-1:2222}
2018-01-08 07:47:35.991077: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job worker -> {0 -> example-job-worker-l7yx-0:2222}
2018-01-08 07:47:35.998491: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:316] Started server with target: grpc://localhost:2222
INFO:root:Finished building server.
INFO:root:Running master.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py:175: initialize_all_variables (fromtensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py:175: initialize_all_variables (fromtensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
INFO:root:Server target: grpc://localhost:2222
2018-01-08 07:47:38.981583: I tensorflow/core/distributed_runtime/master_session.cc:998] Start master session 282bef4931a9408d with config: log_device_placement: true
2018-01-08 07:47:38.992261: I tensorflow/core/common_runtime/simple_placer.cc:872] init: (NoOp)/job:master/replica:0/task:0/cpu:0
2018-01-08 07:47:38.992332: I tensorflow/core/common_runtime/simple_placer.cc:872] Mul_3: (Mul)/job:master/replica:0/task:0/cpu:0
2018-01-08 07:47:38.994043: I tensorflow/core/common_runtime/simple_placer.cc:872] Mul_2: (Mul)/job:ps/replica:0/task:1/cpu:0
2018-01-08 07:47:38.994080: I tensorflow/core/common_runtime/simple_placer.cc:872] Mul_1: (Mul)/job:ps/replica:0/task:0/cpu:0
2018-01-08 07:47:38.994105: I tensorflow/core/common_runtime/simple_placer.cc:872] Mul: (Mul)/job:worker/replica:0/task:0/cpu:0
2018-01-08 07:47:38.994129: I tensorflow/core/common_runtime/simple_placer.cc:872] Const_7: (Const)/job:master/replica:0/task:0/cpu:0
2018-01-08 07:47:38.994150: I tensorflow/core/common_runtime/simple_placer.cc:872] Const_6: (Const)/job:master/replica:0/task:0/cpu:0
2018-01-08 07:47:38.994172: I tensorflow/core/common_runtime/simple_placer.cc:872] Const_5: (Const)/job:ps/replica:0/task:1/cpu:0
2018-01-08 07:47:38.994192: I tensorflow/core/common_runtime/simple_placer.cc:872] Const_4: (Const)/job:ps/replica:0/task:1/cpu:0
2018-01-08 07:47:38.994212: I tensorflow/core/common_runtime/simple_placer.cc:872] Const_3: (Const)/job:ps/replica:0/task:0/cpu:0
2018-01-08 07:47:38.994233: I tensorflow/core/common_runtime/simple_placer.cc:872] Const_2: (Const)/job:ps/replica:0/task:0/cpu:0
2018-01-08 07:47:38.994254: I tensorflow/core/common_runtime/simple_placer.cc:872] Const_1: (Const)/job:worker/replica:0/task:0/cpu:0
2018-01-08 07:47:38.994274: I tensorflow/core/common_runtime/simple_placer.cc:872] Const: (Const)/job:worker/replica:0/task:0/cpu:0
INFO:root:Result: [[   0    1    4    9   16   25   36   49   64   81]
 [ 100  121  144  169  196  225  256  289  324  361]
 [ 400  441  484  529  576  625  676  729  784  841]
 [ 900  961 1024 1089 1156 1225 1296 1369 1444 1521]
 [1600 1681 1764 1849 1936 2025 2116 2209 2304 2401]
 [2500 2601 2704 2809 2916 3025 3136 3249 3364 3481]
 [3600 3721 3844 3969 4096 4225 4356 4489 4624 4761]
 [4900 5041 5184 5329 5476 5625 5776 5929 6084 6241]
 [6400 6561 6724 6889 7056 7225 7396 7569 7744 7921]
 [8100 8281 8464 8649 8836 9025 9216 9409 9604 9801]]
INFO:root:Result: [[   0    1    4    9   16   25   36   49   64   81]
 [ 100  121  144  169  196  225  256  289  324  361]
 [ 400  441  484  529  576  625  676  729  784  841]
 [ 900  961 1024 1089 1156 1225 1296 1369 1444 1521]
 [1600 1681 1764 1849 1936 2025 2116 2209 2304 2401]
 [2500 2601 2704 2809 2916 3025 3136 3249 3364 3481]
 [3600 3721 3844 3969 4096 4225 4356 4489 4624 4761]
 [4900 5041 5184 5329 5476 5625 5776 5929 6084 6241]
 [6400 6561 6724 6889 7056 7225 7396 7569 7744 7921]
 [8100 8281 8464 8649 8836 9025 9216 9409 9604 9801]]
INFO:root:Result: [[   0    1    4    9   16   25   36   49   64   81]
 [ 100  121  144  169  196  225  256  289  324  361]
 [ 400  441  484  529  576  625  676  729  784  841]
 [ 900  961 1024 1089 1156 1225 1296 1369 1444 1521]
 [1600 1681 1764 1849 1936 2025 2116 2209 2304 2401]
 [2500 2601 2704 2809 2916 3025 3136 3249 3364 3481]
 [3600 3721 3844 3969 4096 4225 4356 4489 4624 4761]
 [4900 5041 5184 5329 5476 5625 5776 5929 6084 6241]
 [6400 6561 6724 6889 7056 7225 7396 7569 7744 7921]
 [8100 8281 8464 8649 8836 9025 9216 9409 9604 9801]]
INFO:root:Result: [[   0    1    4    9   16   25   36   49   64   81]
 [ 100  121  144  169  196  225  256  289  324  361]
 [ 400  441  484  529  576  625  676  729  784  841]
 [ 900  961 1024 1089 1156 1225 1296 1369 1444 1521]
 [1600 1681 1764 1849 1936 2025 2116 2209 2304 2401]
 [2500 2601 2704 2809 2916 3025 3136 3249 3364 3481]
 [3600 3721 3844 3969 4096 4225 4356 4489 4624 4761]
 [4900 5041 5184 5329 5476 5625 5776 5929 6084 6241]
 [6400 6561 6724 6889 7056 7225 7396 7569 7744 7921]
 [8100 8281 8464 8649 8836 9025 9216 9409 9604 9801]]
init: (NoOp): /job:master/replica:0/task:0/cpu:0
Mul_3: (Mul): /job:master/replica:0/task:0/cpu:0
Mul_2: (Mul): /job:ps/replica:0/task:1/cpu:0
Mul_1: (Mul): /job:ps/replica:0/task:0/cpu:0
Mul: (Mul): /job:worker/replica:0/task:0/cpu:0
Const_7: (Const): /job:master/replica:0/task:0/cpu:0
Const_6: (Const): /job:master/replica:0/task:0/cpu:0
Const_5: (Const): /job:ps/replica:0/task:1/cpu:0
Const_4: (Const): /job:ps/replica:0/task:1/cpu:0
Const_3: (Const): /job:ps/replica:0/task:0/cpu:0
Const_2: (Const): /job:ps/replica:0/task:0/cpu:0
Const_1: (Const): /job:worker/replica:0/task:0/cpu:0
Const: (Const): /job:worker/replica:0/task:0/cpu:0

```

###Tensorflow Notebook

####Deploy JupyterHub

The second key component of Kubeflow is the ability to run Jupyter Notebooks via JupyterHub. 

Jupyter Notebook is the classic data science tool to run inline scripts and code snippets while documenting the process in the browser.

![pic](http://bos.nj.bpc.baidu.com/v1/agroup/840be8f3fcbebf0bc6e37ce435fc22556c44697d)

```
master $ kubectl get svc
NAME                        TYPE           CLUSTER-IP       EXTERNAL-IP               PORT(S)          AGE
example-job-master-yuks-0   ClusterIP      10.106.249.176   <none>                    2222/TCP         19s
example-job-ps-yuks-0       ClusterIP      10.100.170.233   <none>                    2222/TCP         19s
example-job-ps-yuks-1       ClusterIP      10.97.74.66      <none>                    2222/TCP         19s
example-job-worker-yuks-0   ClusterIP      10.107.161.39    <none>                    2222/TCP         19s
kubernetes                  ClusterIP      10.96.0.1        <none>                    443/TCP          16m
model-service               LoadBalancer   10.106.90.208    <pending>                 9000:31281/TCP   1m
tf-hub-0                    ClusterIP      None             <none>                    8000/TCP         1m
tf-hub-lb                   LoadBalancer   10.103.175.116   172.17.0.59,172.17.0.68   80:31040/TCP     1m
```

To deploy a notebook, a new server has to be started. KubeFlow is using internally the gcr.io/kubeflow/tensorflow-notebook-cpu:v1 Docker Image as default. After accessing the JupyterHub, you can click Start My server button.


The server launcher allows you to configure additional options, such as resource requirements. In this case, accept the defaults and click Spawn to start the server. Now you can see the contents of the Docker image that you can navigate, extend and work with Jupyter Notebooks.

```
master $ kubectl get pods
NAME                               READY     STATUS    RESTARTS   AGE
example-job-ps-yuks-0-4dlbp        1/1       Running   0          26m
example-job-ps-yuks-1-zrm5b        1/1       Running   0          26m
example-job-worker-yuks-0-rqqqt    1/1       Running   0          26m
jupyter-admin                      1/1       Running   0          1m
model-server-584cf76db9-24k2x      1/1       Running   0          27m
model-server-584cf76db9-2m5wj      1/1       Running   0          27m
model-server-584cf76db9-vt567      1/1       Running   0          27m
tf-hub-0                           1/1       Running   0          27m
tf-job-operator-6f7ccdfd4d-t8jtd   1/1       Running   0          27m
```

![pic](http://bos.nj.bpc.baidu.com/v1/agroup/78d0b2713cae879b4e5ddeb49c9266ca0e3625a9)

####Working with Jupyter Notebook

JupyterHub can now be accessed via the pod. You can now work with the environment seamlessly. For example to create a new notebook, select the New dropdown, and select the Python 3 kernel as shown below.

![pic](http://bos.nj.bpc.baidu.com/v1/agroup/d35e8de5a621235808d9a95a1a88497878149fcd)

![pic](http://bos.nj.bpc.baidu.com/v1/agroup/cc05101c7bbe922d6d769d5fabb07ee796841cdd)


###Tensorflow Model Server

####Access Model Server

The final Component is the model server. Once trained, the model can be used to perform predictions for the new data when it's published. By using Kubeflow, it's possible to access the server by deploying jobs to the Kubernetes infrastructure.

####Image Classification

In this example, we use the pre-trained Inception V3 model. It's the architecture trained on ImageNet dataset. The ML task is image classification while the model server and its clients being handled by Kubernetes.

To use the published model, you need to set up the client. This can be achieved the same way as other jobs.

```
master $ cat model-client-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-client-job-katacoda
spec:
  template:
    metadata:
      name: model-client-job-katacoda
    spec:
      containers:
      - name: model-client-job-katacoda
        image: katacoda/tensorflow_serving
        imagePullPolicy: Never
        command:
        - /bin/bash
        - -c
        args:
        - /serving/bazel-bin/tensorflow_serving/example/inception_client
          --server=model-service:9000 --image=/data/katacoda.jpg
        volumeMounts:
        - name: inception-persistent-storage-katacoda
          mountPath: /data
      volumes:
      - name: inception-persistent-storage-katacoda
        hostPath:
          path: /root
      restartPolicy: Never
```
To deploy it use the following command:

```
master $ kubectl get pods -a
NAME                               READY     STATUS      RESTARTS   AGE
example-job-master-r31t-0-f7r2n    0/1       Completed   0          1m
example-job-ps-r31t-0-2s4wf        1/1       Running     0          1m
example-job-ps-r31t-1-7xl64        1/1       Running     0          1m
example-job-worker-r31t-0-pscx9    1/1       Running     0          1m
model-client-job-katacoda-l96kd    1/1       Running     0          12s
model-server-584cf76db9-4t7sr      1/1       Running     0          2m
model-server-584cf76db9-djxfb      1/1       Running     0          2m
model-server-584cf76db9-h7fkq      1/1       Running     0          2m
tf-hub-0                           1/1       Running     0          2m
tf-job-operator-6f7ccdfd4d-7p8d6   1/1       Running     0          2m
```

Output the classification results

```
master $ kubectl logs $(kubectl get pods -a | grep Completed | tail -n1 |  tr -s ' ' | cut -d ' ' -f 1)
D0109 08:08:39.383516352       1 ev_posix.c:101]             Using polling engine: poll
E0109 08:08:46.951275028       1 chttp2_transport.c:1810]    close_transport: {"created":"@1515485326.951245440","description":"FD shutdown","file":"src/core/lib/iomgr/ev_poll_posix.c","file_line":427}
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "comic book"
    string_val: "rubber eraser, rubber, pencil eraser"
    string_val: "coffee mug"
    string_val: "pencil sharpener"
    string_val: "envelope"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 8.31655883789
    float_val: 5.18350791931
    float_val: 4.77944898605
    float_val: 4.31814956665
    float_val: 4.29243946075
  }
}
```



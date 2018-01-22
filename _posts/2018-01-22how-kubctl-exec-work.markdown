##Sends a POST request to the apiserver, to /v1/namespaces/<ns>/pods/<pod>/exec
###Query parameters are used to indicate:
####The command to run
####If stdin should be enabled
####If stdout should be enabled
####If stderr should be enabled
##The name of the container
###The request includes a protocol upgrade from normal HTTP to SPDY
##SPDY allows for separate stdin/stdout/stderr/spdy-error "streams" to be multiplexed over a single TCP connection
##The apiserver establishes a connection to the kubelet for the pod
##The kubelet generates a short-lived token and issues a redirect to the CRI (I'm a bit fuzzy on the details here)
##The CRI handles the exec request and issues a docker exec API call
###The SPDY stdin/stdout/stderr streams are specified as input/output/error to the exec API call

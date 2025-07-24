from rag_pipelines.granite_rag_pipeline import load_granite_rag_pipline
from rag_pipelines.llama31_rag_pipeline import load_llama31_rag_pipeline
from rag_pipelines.llama27_rag_pipeline import load_llama27_rag_pipeline

import json
import time

# UNCOMMENT THE CODE FOR THE MODEL THAT YOU ARE NOT USING BEFORE RUNNING

'''#START OF GRANITE MODEL LOGIC
#granite
graph = load_granite_rag_pipline()
#END OF GRANITE MODEL LOGIC'''


# START OF LLAMA 3.1 LOGIC
#llama 3.1
graph = load_llama31_rag_pipeline()
# END OF LLAMA 3.1 LOGIC'''

'''# START OF LLAMA 2.7 LOGIC
#llama 2.7
graph = load_llama27_rag_pipeline()
# END OF LLAMA 2.7 LOGIC'''


# 30 questions
questions  =[
    "What is the primary objective of the 'Pod Scenarios' feature within Krkn-hub on a Kubernetes/OpenShift cluster?",
    "How can Cerberus be integrated with Krkn-hub's pod scenarios to monitor the cluster and determine the success or failure of a chaos injection?",
    "What is the standard Podman command structure for initiating a pod disruption scenario with Krkn-hub, including how to provide the kube config and enable host environment variables?",
    "After a Krkn-hub pod scenario has been initiated, what commands can be used with both Podman and Docker to monitor its ongoing logs and to retrieve its final exit code for pass/fail determination?",
    "What is a significant limitation of the `--env-host` option when running Krkn-hub scenarios with Podman on certain client types, and how should environment variables be managed in such cases?",
    "Why is it crucial to adjust the permissions of the kube config file before mounting it into the Krkn-hub container, and what specific commands are recommended to achieve this?",
    "How can environment variables be passed to a Krkn-hub scenario container to customize its behavior, detailing both the host-based and command-line methods?",
    "What is the function of the `NAMESPACE` parameter in Krkn-hub pod scenarios, and what is its default value, including its support for advanced matching?",
    "Explain the interplay between the `POD_LABEL` and `NAME_PATTERN` parameters in determining which pods are targeted for disruption, and what their default behaviors are.",
    "What do the `DISRUPTION_COUNT`, `KILL_TIMEOUT`, and `EXPECTED_RECOVERY_TIME` parameters control in a Krkn-hub pod scenario, and what are their default values?",
    "Beyond specific namespace targeting, how can the `NAMESPACE` environment variable be configured to randomly disrupt pods in OpenShift system namespaces, and what additional mode can be enabled for continuous reliability testing?",
    "When `CAPTURE_METRICS` or `ENABLE_ALERTS` are active, how should custom metrics and alerts profiles be provided to the Krkn-hub container, including the specific internal paths for mounting?",
    "When executing a `pod-scenarios` run with `krknctl`, what are the different mechanisms available for precisely identifying the target pods for disruption, and how do they interact?",
    "Describe the two critical timeout parameters in `krknctl run pod-scenarios` and explain their distinct roles in determining the scenario's outcome.",
    "If a user wants to disrupt more than one pod in a `pod-scenarios` run, which parameter needs to be adjusted, and what is the default behavior if this parameter is not explicitly set?",
    "What is the default scope for targeting pods in a `krknctl run pod-scenarios` execution if no specific namespace, label, or name pattern is provided by the user?",
    "How can a user obtain a comprehensive list of all configurable options and their descriptions for the `krknctl run pod-scenarios` command?",
    "Differentiate between the 'Single Pod Deletion' and 'Multiple Pods Deleted Simultaneously' chaos scenarios regarding their simulated failure types and the primary customer impact each aims to validate or expose.",
    "Explain the significance of 'Pod Eviction' as a chaos scenario, detailing its typical triggers and the specific configurations that are crucial for ensuring zero customer impact during such an event.",
    "What specific metrics and observations from Krkn telemetry are crucial for quantitatively assessing the high availability of an application after a chaos engineering run?",
    "Beyond automatic recovery, what are the key architectural and operational indicators, as described in the text, that confirm an application is highly available in a Kubernetes environment?",
    "How does the 'Single Pod Deletion' scenario specifically validate the resilience mechanisms of Kubernetes, and what is the expected recovery timing for stateless applications in this context?",
    "What is the primary concern regarding customer impact when 'Multiple Pods Deleted Simultaneously' occurs, and what architectural characteristic helps mitigate this impact to ensure high availability?",
    "Describe the role of `topologySpreadConstraints` in achieving high availability and how their effective implementation can be verified using standard Kubernetes commands.",
    "In the context of chaos engineering, what does 'Recovery Is Automatic' signify as a high availability indicator, and why is the absence of manual intervention considered critical for true HA?",
    "What is the primary purpose of the `pod_disruption_scenarios` section within a Krkn configuration, and how is a specific scenario file referenced?",
    "How can a user leverage the provided schema file to enhance their experience when creating a Krkn scenario configuration in an IDE?",
    "In a kill-pods chaos scenario targeting Kubernetes components, which component is commonly disrupted and how are its pods typically selected?",
    "In a Krkn pod disruption scenario, what does the `krkn_pod_recovery_time` parameter control, and what is a typical value for it?",
    "Beyond a general 'basic pod scenario,' what are some of the critical Kubernetes or OpenShift components for which Krkn offers pre-defined chaos scenarios, and are they currently functional?",
    "If a user wanted to specifically target the Kubernetes API server for a chaos experiment, which Krkn scenario would be appropriate, and what action would it perform?",
    "How does the 'OpenShift System Pods' chaos scenario differ in its targeting approach compared to component-specific scenarios like 'Etcd' or 'Prometheus'?",
    "What is the purpose of the `id` field within a Krkn scenario configuration?",
    "Can the `kill-pods` scenario be configured to disrupt pods outside of the `kube-system` namespace?",
    "Describe the hierarchical structure for defining chaos scenarios within the top-level `kraken` configuration block."
  ]
reference_answers = [
    "The primary objective of the 'Pod Scenarios' feature in Krkn-hub is to disrupt pods that match a specified label within a designated namespace on either a Kubernetes or OpenShift cluster, simulating chaos conditions.",
    "To integrate Cerberus, it must be started before injecting chaos. For the chaos injection container to auto-connect with Cerberus and enable monitoring for pass/fail evaluation, the `CERBERUS_ENABLED` environment variable must be set.",
    "The standard Podman command is `podman run --name=<container_name> --net=host --env-host=true -v <path-to-kube-config>:/home/krkn/.kube/config:Z -d containers.krkn-chaos.dev/krkn-chaos/krkn-hub:pod-scenarios`. This command runs the scenario container, mounts the kube config, and uses `--env-host=true` to allow the container to access host environment variables.",
    "To monitor ongoing logs, use `podman logs -f <container_name or container_id>` or `docker logs -f <container_name or container_id>`. To determine the final outcome (pass/fail), the exit code can be retrieved using `podman inspect <container-name or container_id> --format \"{{.State.ExitCode}}\"` or `docker inspect <container-name or container_id> --format \"{{.State.ExitCode}}\"`.",
    "The `--env-host` option is not available with remote Podman clients, including those on Mac and Windows (excluding WSL2). In these situations, environment variables must be set individually on the Podman command line using the `-e <VARIABLE>=<value>` syntax for each variable.",
    "It is crucial because the Krkn-hub container runs with a non-root user, requiring the kube config to be globally readable. This can be achieved by first flattening the config and saving it (`kubectl config view --flatten > ~/kubeconfig`), and then changing its permissions to `444` (`chmod 444 ~/kubeconfig`) before mounting.",
    "Environment variables can be passed by setting them on the host running the container using `export <parameter_name>=<value>` if the `--env-host` option is used. Alternatively, they can always be passed directly on the command line when running the container using the `-e <VARIABLE>=<value>` flag.",
    "The `NAMESPACE` parameter specifies the targeted namespace(s) within the cluster where the pod disruption will occur. It supports regular expressions for flexible matching, and its default value is `openshift-.*`, which targets all OpenShift system namespaces.",
    "The `POD_LABEL` parameter specifies the label of the pod(s) to target, with a default of an empty string. If `POD_LABEL` is not specified, the `NAME_PATTERN` parameter is used instead to match pods within the `NAMESPACE` using a regex pattern. The `NAME_PATTERN` defaults to `.*`, meaning it will match all pods in the targeted namespace if no specific label is provided.",
    "The `DISRUPTION_COUNT` parameter specifies the number of pods to disrupt, defaulting to `1`. The `KILL_TIMEOUT` parameter sets the maximum time (in seconds) to wait for target pods to be removed, defaulting to `180`. The `EXPECTED_RECOVERY_TIME` parameter defines the timeout (in seconds) for disrupted pods to recover, defaulting to `120`, and the scenario fails if recovery doesn't occur within this period.",
    "The `NAMESPACE` environment variable can be set to `openshift-.*` to randomly select and disrupt pods within OpenShift system namespaces. For continuous reliability testing, `DAEMON_MODE` can be enabled to disrupt pods every 'x' seconds in the background.",
    "When `CAPTURE_METRICS` or `ENABLE_ALERTS` are active, custom profiles must be mounted from the host into the container using volume mounts. The custom metrics profile should be mounted to `/home/krkn/kraken/config/metrics-aggregated.yaml`, and the custom alerts profile to `/home/krkn/kraken/config/alerts`.",
    "To precisely identify target pods for disruption, users can leverage three parameters: `--namespace`, `--pod-label`, and `--name-pattern`. The `--namespace` parameter allows targeting pods within a specific namespace, supporting regular expressions, with a default of `openshift-*`. The `--pod-label` parameter targets pods based on a specific label, such as \"app=test\". If `--pod-label` is not specified, the `--name-pattern` parameter, which defaults to `.*`, is used as a regex pattern to match pods within the specified `--namespace`.",
    "The two critical timeout parameters are `--kill-timeout` and `--expected-recovery-time`. The `--kill-timeout` parameter, defaulting to 180 seconds, specifies the maximum time to wait for the target pod(s) to be removed from the cluster. In contrast, the `--expected-recovery-time` parameter, defaulting to 120 seconds, dictates the maximum time allowed for a disrupted pod to recover. If a pod fails to recover within this `expected-recovery-time`, the entire scenario will be marked as a failure.",
    "To disrupt more than one pod, the user needs to adjust the `--disruption-count` parameter. By default, if this parameter is not explicitly set, `krknctl run pod-scenarios` will only disrupt a single pod, as its default value is 1.",
    "If no specific targeting parameters are provided, the `krknctl run pod-scenarios` execution will default to targeting pods within any namespace matching the `openshift-*` regex pattern, as specified by the `--namespace` parameter's default. Furthermore, within these namespaces, it will match any pod name due to the `--name-pattern` defaulting to `.*` when `--pod-label` is not specified.",
    "A user can obtain a comprehensive list of all available scenario options and their descriptions for the `krknctl run pod-scenarios` command by executing `krknctl run pod-scenarios --help`.",
    "The 'Single Pod Deletion' scenario simulates an unplanned, isolated failure of a single pod, primarily validating whether the ReplicaSet or Deployment automatically creates a replacement to ensure continuous service with minimal customer impact. In contrast, 'Multiple Pods Deleted Simultaneously' simulates a larger, more widespread failure event, such as a node crash or an Availability Zone outage. This scenario tests the system's ability to recover gracefully with sufficient resources and policies, as a failure of all pods belonging to a service in this scenario directly impacts user experience if not properly mitigated by distributed replicas.",
    "Pod Eviction is a significant chaos scenario because it simulates a 'soft disruption' triggered by Kubernetes itself, typically during routine operations like node upgrades or scaling down. Its importance lies in ensuring graceful termination and restart of pods elsewhere without user impact. For zero customer impact, it is crucial that readiness and liveness probes are correctly configured to manage pod health and traffic flow, and that Pod Disruption Budgets (PDBs) are in place to define the minimum number of available replicas, thereby ensuring that a rolling disruption does not take down the entire application.",
    "Krkn telemetry provides vital end-of-run data that is crucial for quantitatively assessing an application's high availability. Key indicators include recovery times, which measure how quickly services are restored post-disruption; pod reschedule latency, indicating the time taken for new pods to become ready and operational; and service downtime, which quantifies any periods of unavailability experienced by the application. These metrics collectively offer a precise measure of the system's resilience and its ability to recover automatically.",
    "To confirm an application's high availability in a Kubernetes environment, several architectural and operational indicators are crucial. Architecturally, it's essential to verify that multiple replicas exist for deployments (e.g., `kubectl get deploy` showing >1 replicas) and that pods are distributed across different nodes or availability zones, often achieved using `topologySpreadConstraints` or observed via `kubectl get pods -o wide`. Operationally, during chaos tests, service uptime must remain unaffected, which can be verified through synthetic probes or Prometheus alerts, and crucially, recovery from disruptions must be entirely automatic, requiring no manual intervention.",
    "The 'Single Pod Deletion' scenario specifically validates whether the ReplicaSet or Deployment automatically creates a replacement pod when an individual pod is unexpectedly deleted. This mechanism ensures continuous service even if a single pod crashes. For stateless applications, the expected recovery timing in this scenario is typically less than 10 seconds, a performance metric often observed and confirmed through Krkn telemetry output.",
    "The primary concern regarding customer impact when 'Multiple Pods Deleted Simultaneously' occurs is that if all pods of a service fail, the user experience is directly impacted, potentially leading to service unavailability. This significant impact can be mitigated and high availability ensured if the application is designed with redundancy, allowing it to continue functioning from other replicas that are distributed across different zones or nodes, preventing a single point of failure from taking down the entire service.",
    "`topologySpreadConstraints` play a crucial role in achieving high availability by ensuring that pods are distributed evenly across different failure domains, such as nodes or availability zones. This prevents a single point of failure from impacting all replicas of an application. The effective implementation and resulting distribution can be observed using standard Kubernetes commands like `kubectl get pods -o wide`, which displays the node on which each pod is running, allowing an administrator to visually confirm the desired spread across the cluster's topology.",
    "In chaos engineering, 'Recovery Is Automatic' as a high availability indicator signifies that the system is designed to self-heal and restore service without any human intervention after a disruption. The absence of manual intervention is considered critical for true HA because it demonstrates the system's inherent resilience and automation capabilities. Manual intervention introduces delays, increases the potential for human error, and indicates a dependency on operational teams rather than robust, self-managing infrastructure, undermining the goal of continuous service availability.",
    "The `pod_disruption_scenarios` section in a Krkn configuration is used to define and include specific pod chaos scenarios. A scenario file is referenced by providing its `path/to/scenario.yaml` within this section, allowing Krkn to execute the disruption defined in that file.",
    "Users can enhance their experience by adjusting the schema reference in their scenario file to point to the `plugin.schema.json` file (e.g., `yaml-language-server: $schema=../plugin.schema.json`). This integration provides valuable features like code completion and documentation for available options directly within their IDE.",
    "The `kill-pods` scenario targets the `kube-scheduler` component. It selects pods by looking for those within the `kube-system` namespace (specified by `namespace_pattern: ^kube-system$`) that also possess the label `k8s-app=kube-scheduler` (specified by `label_selector`).",
    "The `krkn_pod_recovery_time` parameter specifies the duration, in seconds, that Krkn will wait for a disrupted pod to recover. A commonly used value is `120`, which represents a 120‑second recovery period.",
    "Krkn offers pre-defined chaos scenarios for critical components such as Etcd, Kube ApiServer, ApiServer, Prometheus, and random pods running in OpenShift system namespaces. These scenarios are listed as functional in the Krkn documentation.",
    "To target the Kubernetes API server, the 'Kube ApiServer' scenario would be appropriate. This scenario is designed to kill either a single or multiple replicas of the kube-apiserver component.",
    "The 'OpenShift System Pods' scenario targets random pods running across various OpenShift system namespaces, providing a broader, less specific disruption. In contrast, component-specific scenarios like 'Etcd' or 'Prometheus' precisely target replicas of those named components, focusing disruption on a particular service.",
    "The `id` field, serves as a unique identifier for a specific chaos scenario defined within the Krkn configuration file. It provides a distinct name for the scenario being executed.",
    "By default, the `kill-pods` scenario is restricted to the `kube-system` namespace through the `namespace_pattern: ^kube-system$` setting. However, this setting can be modified to target pods in other namespaces if desired.",
    "Within the `kraken` configuration block, chaos scenarios are organized under the `chaos_scenarios` key. This key contains a list, where each item is a `pod_disruption_scenarios` entry, which in turn specifies the `path/to/scenario.yaml` file containing the detailed configuration for a particular pod disruption."
  ]

evaluation_data = []

for i, q in enumerate(questions):
    start_time = time.time()
    result = graph.invoke({"question": q})
    end_time = time.time()
    duration = end_time - start_time

    retrieved_context = [doc.page_content for doc in result["context"]]
    
    if i < len(reference_answers):
        evaluation_data.append({
            "user_input": q,
            "generated_answer": result["answer"],
            "retrieved_context": retrieved_context,
            "reference_answer": reference_answers[i],
            "duration_seconds": duration      #not a part of the json structure but useful metrics to have
        })


output = {
    "items": evaluation_data,
    "email": "user_email@example.com"  
}

# Save to file
with open("rag_evaluation_data.json", "w") as f:
    json.dump(output, f, indent=2)

print("Evaluation data saved to 'rag_evaluation_data.json'")

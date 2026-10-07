import re


KUBERNETES_KEYWORDS = {
    "kubernetes",
    "k8s",
    "kubectl",
    "pod",
    "pods",
    "deployment",
    "deployments",
    "service",
    "services",
    "namespace",
    "namespaces",
    "configmap",
    "secret",
    "ingress",
    "gateway",
    "httproute",
    "role",
    "rolebinding",
    "clusterrole",
    "clusterrolebinding",
    "serviceaccount",
    "rbac",
    "node",
    "nodes",
    "daemonset",
    "statefulset",
    "ns",
    "replicaset",
    "cronjob",
    "job",
    "helm",
    "etcd",
    "kubelet",
    "kube-proxy",
    "kube-apiserver",
    "containerd",
    "crictl",
    "nodeport",
    "image",
}


SENSITIVE_PATTERNS = {
    "AWS Access Key": r"\bAKIA[0-9A-Z]{16}\b",

    "Private Key": (
        r"-----BEGIN "
        r"(?:RSA |EC |OPENSSH )?"
        r"PRIVATE KEY-----"
    ),

    "Bearer Token": (
        r"(?i)\bbearer\s+"
        r"[A-Za-z0-9\-._~+/]+=*\b"
    ),

    "Generic API Key": (
        r"(?i)\b(api[_-]?key|apikey|access[_-]?token)"
        r"\s*[:=]\s*['\"]?"
        r"[A-Za-z0-9_\-]{16,}"
    ),

    "Password": (
        r"(?i)\b(password|passwd|pwd)"
        r"\s*[:=]\s*['\"]?"
        r"[^\s'\"]{6,}"
    ),

    "GitHub Token": (
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"
    ),
}


def contains_sensitive_data(text: str):
    """
    Returns:
        (True, type)  -> sensitive data detected
        (False, None) -> safe
    """
    for name, pattern in SENSITIVE_PATTERNS.items():
        if re.search(pattern, text):
            return True, name

    return False, None


def is_kubernetes_related(text: str) -> bool:
    text = text.lower()

    return any(
        re.search(rf"\b{re.escape(keyword)}\b", text)
        for keyword in KUBERNETES_KEYWORDS
    )


def guardrail(user_message: str):
    # FIRST: sensitive-data check
    sensitive, sensitive_type = contains_sensitive_data(user_message)

    if sensitive:
        return {
            "allowed": False,
            "reason": f"Sensitive data detected: {sensitive_type}",
            "send_to_llm": False,
        }

    # SECOND: Kubernetes domain check
    if not is_kubernetes_related(user_message):
        return {
            "allowed": False,
            "reason": "Only Kubernetes-related questions are allowed.",
            "send_to_llm": False,
        }

    return {
        "allowed": True,
        "reason": "Request passed guardrails.",
        "send_to_llm": True,
    }
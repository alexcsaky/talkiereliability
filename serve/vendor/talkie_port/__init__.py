"""Vendored copy of the community transformers port of Talkie (Apache 2.0).

Loaded directly instead of through ``trust_remote_code``, so no code from the
hub runs in our containers. See README.md in this folder for provenance.
"""
from .configuration_talkie import TalkieConfig
from .modeling_talkie import TalkieForCausalLM, TalkieModel

__all__ = ["TalkieConfig", "TalkieForCausalLM", "TalkieModel"]

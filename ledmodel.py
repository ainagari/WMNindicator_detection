from __future__ import annotations

from transformers.models.led.modeling_led import LEDPreTrainedModel, LEDConfig, LEDEncoder, LEDClassificationHead, LEDModel, LEDSeq2SeqSequenceClassifierOutput, LEDEncoderBaseModelOutput
import torch
from torch import nn
import warnings
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union



import json
import logging
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

import huggingface_hub
from transformers import AutoConfig, AutoModel, AutoTokenizer, MT5Config, PretrainedConfig, T5Config
from transformers.utils.import_utils import is_peft_available
from transformers.utils.peft_utils import find_adapter_config_file



from transformers.activations import ACT2FN
logger = logging.getLogger(__name__)


from transformers.modeling_outputs import MaskedLMOutput

#if TYPE_CHECKING and is_peft_available():
#    from peft import PeftConfig



class ModifiedTransformer(nn.Module):
    """Hugging Face AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    """


    save_in_root: bool = True

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int | None = None,
        model_args: dict[str, Any] | None = None,
        tokenizer_args: dict[str, Any] | None = None,
        config_args: dict[str, Any] | None = None,
        cache_dir: str | None = None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: str = None,
        backend: str = "torch",
    ) -> None:
        super().__init__()
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case
        self.backend = backend
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        if config_args is None:
            config_args = {}

        config, is_peft_model = self._load_config(model_name_or_path, cache_dir, backend, config_args)
        self._load_model(model_name_or_path, config, cache_dir, backend, is_peft_model, **model_args)

        if max_seq_length is not None and "model_max_length" not in tokenizer_args:
            tokenizer_args["model_max_length"] = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if (
                hasattr(self.auto_model, "config")
                and hasattr(self.auto_model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def _load_config(self, model_name_or_path, cache_dir, backend, config_args):
    #) -> tuple[PeftConfig | PretrainedConfig, bool]:
        #self, model_name_or_path: str, cache_dir: str | None, backend: str, config_args: dict[str, Any]
    #) -> tuple[PeftConfig | PretrainedConfig, bool]:
        """Loads the transformers or PEFT configuration

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
                or the path to a local model directory.
            cache_dir (str | None): The cache directory to store the model configuration.
            backend (str): The backend used for model inference. Can be `torch`, `onnx`, or `openvino`.
            config_args (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers config.

        Returns:
            tuple[PretrainedConfig, bool]: The model configuration and a boolean indicating whether the model is a PEFT model.
        """
        if (
            find_adapter_config_file(
                model_name_or_path,
                cache_dir=cache_dir,
                token=config_args.get("token"),
                revision=config_args.get("revision"),
                local_files_only=config_args.get("local_files_only", False),
            )
            is not None
        ):
            if not is_peft_available():
                raise Exception(
                    "Loading a PEFT model requires installing the `peft` package. You can install it via `pip install peft`."
                )
            if backend != "torch":
                # TODO: Consider following these steps automatically so we can load PEFT models with other backends
                raise ValueError(
                    "PEFT models can currently only be loaded with the `torch` backend. "
                    'To use other backends, load the model with `backend="torch"`, call `model[0].auto_model.merge_and_unload()`, '
                    "save that model with `model.save_pretrained()` and then load the model with the desired backend."
                )
            from peft import PeftConfig

            return PeftConfig.from_pretrained(model_name_or_path, **config_args, cache_dir=cache_dir), True

        return AutoConfig.from_pretrained(model_name_or_path, **config_args, cache_dir=cache_dir), False

    def _load_model(
        self,
        model_name_or_path: str,
        config: PeftConfig | PretrainedConfig,
        cache_dir: str,
        backend: str,
        is_peft_model: bool,
        **model_args,
    ) -> None:
        """Loads the transformers or PEFT model into the `auto_model` attribute

        Args:
            model_name_or_path (str): The model name on Hugging Face (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
                or the path to a local model directory.
            config ("PeftConfig" | PretrainedConfig): The model configuration.
            cache_dir (str | None): The cache directory to store the model configuration.
            backend (str): The backend used for model inference. Can be `torch`, `onnx`, or `openvino`.
            is_peft_model (bool): Whether the model is a PEFT model.
            model_args (dict[str, Any]): Keyword arguments passed to the Hugging Face Transformers model.
        """
        if backend == "torch":
            # When loading a PEFT model, we need to load the base model first,
            # but some model_args are only for the adapter
            adapter_only_kwargs = {}
            if is_peft_model:
                for adapter_only_kwarg in ["revision"]:
                    if adapter_only_kwarg in model_args:
                        adapter_only_kwargs[adapter_only_kwarg] = model_args.pop(adapter_only_kwarg)

            if isinstance(config, T5Config):
                self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
            elif isinstance(config, MT5Config):
                self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
            elif isinstance(config, LEDConfig):
                self._load_LED_model(model_name_or_path, config, cache_dir, **model_args)
            else:
                self.auto_model = AutoModel.from_pretrained(
                    model_name_or_path, config=config, cache_dir=cache_dir, **model_args
                )


        else:
            raise ValueError(f"Unsupported backend '{backend}'. `backend` should be `torch`, `onnx`, or `openvino`.")


    def _backend_should_export(
        self,
        load_path: Path,
        is_local: bool,
        model_args: dict[str, Any],
        target_file_name: str,
        target_file_glob: str,
        backend_name: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Determines whether the model should be exported to the backend, or if it can be loaded directly.
        Also update the `file_name` and `subfolder` model_args if necessary.

        These are the cases:

        1. If export is set in model_args, just return export
        2. If `<subfolder>/<file_name>` exists; set export to False
        3. If `<backend>/<file_name>` exists; set export to False and set subfolder to the backend (e.g. "onnx")
        4. If `<file_name>` contains a folder, add those folders to the subfolder and set the file_name to the last part

        We will warn if:

        1. The expected file does not exist in the model directory given the optional file_name and subfolder.
           If there are valid files for this backend, but they're don't align with file_name, then we give a useful warning.
        2. Multiple files are found in the model directory that match the target file name and the user did not
           specify the desired file name via `model_kwargs={"file_name": "<file_name>"}`

        Args:
            load_path: The model repository or directory, as a Path instance
            is_local: Whether the model is local or remote, i.e. whether load_path is a local directory
            model_args: The model_args dictionary. Notable keys are "export", "file_name", and "subfolder"
            target_file_name: The expected file name in the model directory, e.g. "model.onnx" or "openvino_model.xml"
            target_file_glob: The glob pattern to match the target file name, e.g. "*.onnx" or "openvino*.xml"
            backend_name: The human-readable name of the backend for use in warnings, e.g. "ONNX" or "OpenVINO"

        Returns:
            Tuple[bool, dict[str, Any]]: A tuple of the export boolean and the updated model_args dictionary.
        """

        export = model_args.pop("export", None)
        if export:
            return export, model_args

        file_name = model_args.get("file_name", target_file_name)
        subfolder = model_args.get("subfolder", None)
        primary_full_path = Path(subfolder, file_name).as_posix() if subfolder else Path(file_name).as_posix()
        secondary_full_path = (
            Path(subfolder, self.backend, file_name).as_posix()
            if subfolder
            else Path(self.backend, file_name).as_posix()
        )
        glob_pattern = f"{subfolder}/**/{target_file_glob}" if subfolder else f"**/{target_file_glob}"

        # Get the list of files in the model directory that match the target file name
        if is_local:
            model_file_names = [path.relative_to(load_path).as_posix() for path in load_path.glob(glob_pattern)]
        else:
            all_files = huggingface_hub.list_repo_files(
                load_path.as_posix(),
                repo_type="model",
                revision=model_args.get("revision", None),
                token=model_args.get("token", None),
            )
            model_file_names = [fname for fname in all_files if fnmatch(fname, glob_pattern)]

        # First check if the expected file exists in the root of the model directory
        # If it doesn't, check if it exists in the backend subfolder.
        # If it does, set the subfolder to include the backend
        model_found = primary_full_path in model_file_names
        if not model_found and "subfolder" not in model_args:
            model_found = secondary_full_path in model_file_names
            if model_found:
                if len(model_file_names) > 1 and "file_name" not in model_args:
                    logger.warning(
                        f"Multiple {backend_name} files found in {load_path.as_posix()!r}: {model_file_names}, defaulting to {secondary_full_path!r}. "
                        f'Please specify the desired file name via `model_kwargs={{"file_name": "<file_name>"}}`.'
                    )
                model_args["subfolder"] = self.backend
                model_args["file_name"] = file_name
        if export is None:
            export = not model_found

        # If the file_name contains subfolders, set it as the subfolder instead
        file_name_parts = Path(file_name).parts
        if len(file_name_parts) > 1:
            model_args["file_name"] = file_name_parts[-1]
            model_args["subfolder"] = Path(model_args.get("subfolder", ""), *file_name_parts[:-1]).as_posix()

        if export:
            logger.warning(
                f"No {file_name!r} found in {load_path.as_posix()!r}. Exporting the model to {backend_name}."
            )

            if model_file_names:
                logger.warning(
                    f"If you intended to load one of the {model_file_names} {backend_name} files, "
                    f'please specify the desired file name via `model_kwargs={{"file_name": "{model_file_names[0]}"}}`.'
                )

        return export, model_args

    def _backend_warn_to_save(self, model_name_or_path: str, is_local: str, backend_name: str) -> None:
        to_log = f"Saving the exported {backend_name} model is heavily recommended to avoid having to export it again."
        if is_local:
            to_log += f" Do so with `model.save_pretrained({model_name_or_path!r})`."
        else:
            to_log += f" Do so with `model.push_to_hub({model_name_or_path!r}, create_pr=True)`."
        logger.warning(to_log)


    def _load_LED_model(self, model_name_or_path: str, config: PretrainedConfig, cache_dir: str, **model_args) -> None:
        self.auto_model = ModifiedLEDModel.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )


    def __repr__(self) -> str:
        return f"Transformer({self.get_config_dict()}) with Transformer model: {self.auto_model.__class__.__name__} "

    def forward(self, features: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Returns token_embeddings, cls_token"""
        trans_features = {
            key: value
            for key, value in features.items()
            if key in ["input_ids", "attention_mask", "token_type_ids", "inputs_embeds"]
        }

        output_states = self.auto_model(**trans_features, **kwargs, return_dict=False)
        output_tokens = output_states[0]

        # If the AutoModel is wrapped with a PeftModelForFeatureExtraction, then it may have added virtual tokens
        # We need to extend the attention mask to include these virtual tokens, or the pooling will fail
        if is_peft_available():
            from peft import PeftModelForFeatureExtraction

            if (
                isinstance(self.auto_model, PeftModelForFeatureExtraction)
                and self.auto_model.active_peft_config.is_prompt_learning
            ):
                batch_size = output_tokens.size(0)
                attention_mask = features["attention_mask"]
                prefix_attention_mask = torch.ones(
                    batch_size, self.auto_model.active_peft_config.num_virtual_tokens, device=attention_mask.device
                )
                features["attention_mask"] = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        features["token_embeddings"] = output_tokens

        if self.auto_model.config.output_hidden_states and len(output_states) > 2:
            all_layer_idx = 2  # I.e. after last_hidden_states and pooler_output
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features["all_layer_embeddings"] = hidden_states

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

    def tokenize(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: str | bool = True
    ) -> dict[str, torch.Tensor]:
        """Tokenizes a text and maps tokens to token-ids"""
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer(
                *to_tokenize,
                padding=padding,
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )
        )
        return output

    def get_config_dict(self) -> dict[str, Any]:
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str, safe_serialization: bool = True) -> None:
        self.auto_model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, "sentence_bert_config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @classmethod
    def load(cls, input_path: str) -> ModifiedTransformer:
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        # Don't allow configs to set trust_remote_code
        if "model_args" in config and "trust_remote_code" in config["model_args"]:
            config["model_args"].pop("trust_remote_code")
        if "tokenizer_args" in config and "trust_remote_code" in config["tokenizer_args"]:
            config["tokenizer_args"].pop("trust_remote_code")
        if "config_args" in config and "trust_remote_code" in config["config_args"]:
            config["config_args"].pop("trust_remote_code")
        return cls(model_name_or_path=input_path, **config)


class ModifiedLEDModel(LEDPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: LEDConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = LEDEncoder(config, self.shared)
        #self.decoder = LEDDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        #self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    #def get_decoder(self):
    #    return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        #decoder_input_ids: Optional[torch.LongTensor] = None,
        #decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        #decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        #decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], LEDEncoderBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Using this like Bart, as LED is derived from it. So far
        # No checkpoint on the hub exists that uses that in practice.
        # https://github.com/huggingface/transformers/blob/ac3cb660cad283163f7c73cad511124e845ca388/src/transformers/models/bart/modeling_bart.py#L1153
        #if decoder_input_ids is None and decoder_inputs_embeds is None:
        #    decoder_input_ids = shift_tokens_right(
        #        input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        #    )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a LEDEncoderBaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, LEDEncoderBaseModelOutput):
            encoder_outputs = LEDEncoderBaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                global_attentions=encoder_outputs[3] if len(encoder_outputs) > 3 else None,
            )

        #if not return_dict:
        return encoder_outputs





class ModifiedLEDForSequenceClassification(LEDPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: LEDConfig, **kwargs):
        warnings.warn(
            "The `transformers.LEDForSequenceClassification` class is deprecated and will be removed in version 5 of"
            " Transformers. No actual method were provided in the original paper on how to perfom"
            " sequence classification.",
            FutureWarning,
        )
        super().__init__(config, **kwargs)
        self.led = LEDModel(config)
        #encoder = self.led.encoder
        #self.encoder = LEDModel(config).encoder #.self.led.encoder
        #self.encoder = LEDModel(config).encoder #encoder
        self.classification_head = LEDClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        #decoder_input_ids: Optional[torch.LongTensor] = None,
        #decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        #decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        #decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], LEDSeq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        encoder_outputs = self.led.encoder(input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        hidden_states = encoder_outputs[0]  # last hidden state

        eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
            :, -1, :
        ]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return LEDSeq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            #past_key_values=encoder_outputs.past_key_values,
            #decoder_hidden_states=encoder_outputs.decoder_hidden_states,
            #decoder_attentions=encoder_outputs.decoder_attentions,
            #cross_attentions=encoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state, #encoder_last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,#encoder_hidden_states,
            encoder_attentions=encoder_outputs.attentions, #encoder_attentions,
            encoder_global_attentions=encoder_outputs.global_attentions, #encoder_global_attentions,
        )


class LEDPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states




class LEDLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LEDPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states




class LEDOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = LEDLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores



class ModifiedLEDForMLM(LEDPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: LEDConfig, **kwargs):
        warnings.warn(
            "The `transformers.LEDForSequenceClassification` class is deprecated and will be removed in version 5 of"
            " Transformers. No actual method were provided in the original paper on how to perfom"
            " sequence classification.",
            FutureWarning,
        )
        super().__init__(config, **kwargs)

        config.hidden_act = "gelu"
        config.layer_norm_eps = 1e-12
        self.led = LEDModel(config)


        ### HERE: put the MLM head based on a combination of 
        #bertonlymlmhead and ledclassificationhead.
        self.cls = LEDOnlyMLMHead(config)

        #encoder = self.led.encoder
        #self.encoder = LEDModel(config).encoder #.self.led.encoder
        #self.encoder = LEDModel(config).encoder #encoder
        self.classification_head = LEDClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        #decoder_input_ids: Optional[torch.LongTensor] = None,
        #decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        #decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        #decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], LEDSeq2SeqSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        encoder_outputs = self.led.encoder(input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        hidden_states = encoder_outputs[0]  # last hidden state

        #eos_mask = input_ids.eq(self.config.eos_token_id).to(hidden_states.device)

        #if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
        #    raise ValueError("All examples must have the same number of <eos> tokens.")
        sequence_output = hidden_states 
        
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        )







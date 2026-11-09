import copy
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.grpo import validate_native
from test_grpo_v2 import Tokenizer,native,SAMPLING


def test_real_captured_151643_stop_uses_frozen_model_eos(tmp_path):
    fixture=json.loads((ROOT/'tests/fixtures/grpo_native_stop_151643.json').read_text(encoding='utf8'))
    (tmp_path/'generation_config.json').write_text(json.dumps(fixture['generation_config']))
    class CapturedTokenizer:
        eos_token_id=151645
        name_or_path=str(tmp_path)
        def apply_chat_template(self,messages,**kwargs):
            assert messages==fixture['messages']
            return fixture['generation']['prompt_token_ids']
        def decode(self,ids,**kwargs):
            assert ids==fixture['generation']['response_token_ids']
            return fixture['generation']['text']
    assert fixture['generation']['response_token_ids'][-1]==151643
    validate_native(CapturedTokenizer(),fixture['messages'],fixture['generation'],fixture['sampling'])


def tokenizer_with_config(tmp_path,eos):
    (tmp_path/'generation_config.json').write_text(json.dumps({'eos_token_id':eos}))
    tok=Tokenizer(); tok.name_or_path=str(tmp_path)
    return tok


def test_all_configured_native_eos_are_accepted(tmp_path):
    tok=tokenizer_with_config(tmp_path,[2,7])
    for eos in [2,7]: validate_native(tok,[],native([4,eos],'stop'),SAMPLING)
    with pytest.raises(ValueError,match='terminal EOS'):
        validate_native(tok,[],native([4,6],'stop'),SAMPLING)
    with pytest.raises(ValueError,match='terminal EOS'):
        validate_native(tok,[],native([7,4,2],'stop'),SAMPLING)


def test_length_cap_takes_priority_over_final_eos(tmp_path):
    tok=tokenizer_with_config(tmp_path,[2,7]); s={**SAMPLING,'max_new_tokens':3}
    for eos in [2,7,6]: validate_native(tok,[],native([4,5,eos],'length'),s)
    for ids in [[7,5,6],[4,7,2],[4,7]]:
        with pytest.raises(ValueError): validate_native(tok,[],native(ids,'length'),s)


@pytest.mark.parametrize('eos',[[2,True],[],['7'],None])
def test_malformed_model_eos_is_rejected(tmp_path,eos):
    tok=tokenizer_with_config(tmp_path,eos)
    with pytest.raises(ValueError,match='model EOS'):
        validate_native(tok,[],native([4,2],'stop'),SAMPLING)


def test_no_model_config_keeps_primary_eos_only(tmp_path):
    tok=Tokenizer(); tok.name_or_path=str(tmp_path)
    validate_native(tok,[],native([4,2],'stop'),SAMPLING)
    with pytest.raises(ValueError): validate_native(tok,[],native([4,7],'stop'),SAMPLING)

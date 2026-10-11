from pathlib import Path
import sys,json
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from medsim.clinical.config import load_clinical_config
from medsim.clinical.agent_npc import ConsultantAgent
class OfflineClient:
    def is_available(self):return True
checks=[]
for profile,expected in [('mdt_qwen.yaml','qwen-turbo'),('mdt_kimi.yaml','kimi-k2.5'),('strong_primary_mdt.yaml','glm-5')]:
    cfg=load_clinical_config(ROOT/'configs/paper'/profile);calls=[]
    def complete(client,model,*args,**kwargs):
        calls.append(model)
        return {'advice_action':'ASK','utterance':'fixture','recommended_question':['fixture']}
    with patch('medsim.clinical.agent_npc._complete_json_strict',complete):
        ConsultantAgent(None,cfg,OfflineClient()).review_case(turn_index=2,question_count=1,test_count=0,released_results=[],tests_requested=[],context_prompt='offline fixture')
    assert calls==[expected],(profile,calls,expected)
    checks.append({'profile':profile,'consultant':calls[0]})
print(json.dumps({'status':'passed','api_calls':0,'checks':checks}))

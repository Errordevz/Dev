import test from 'node:test';
import assert from 'node:assert/strict';
import {env} from '@huggingface/transformers';
import {providerConfig} from './agent.js';

test('transformers runtime is installed',()=>{
  assert.equal(typeof env.cacheDir,'string');
});

test('provider config exposes standalone local model',()=>{
  const c=providerConfig();
  assert.equal(typeof c.configured,'boolean');
  assert.equal(typeof c.url,'string');
  assert.equal(c.localModel.id,'onnx-community/SmolLM2-135M-Instruct-ONNX-MHA');
  assert.equal(c.localModel.dtype,'q4f16');
});

import {evaluateDecision} from './decision.js';

test('typed decision fallback supports choice, score, and noul', async()=>{
  const choice=await evaluateDecision({
    type:'choice',
    state:'Use the simplest JavaScript implementation with no new dependency.',
    question:'Which approach should be selected?',
    options:['simple JavaScript','large new framework']
  });
  assert.equal(typeof choice.selected,'string');
  assert.equal(Object.keys(choice.probabilities).length,2);
  assert.ok(choice.confidence>=0 && choice.confidence<=1);

  const score=await evaluateDecision({
    type:'score',
    state:'The implementation has tests and clear rollback steps.',
    question:'How prepared is the change?',
    rungs:[
      {value:0,label:'not prepared'},
      {value:1,label:'partly prepared'},
      {value:2,label:'well prepared'}
    ]
  });
  assert.ok([0,1,2].includes(score.selected));

  const noul=await evaluateDecision({
    type:'noul',
    state:'The change has not been verified yet and the rollback path is unknown.',
    question:'Is the change ready for irreversible execution?'
  });
  assert.equal(typeof noul.selected,'boolean');
});

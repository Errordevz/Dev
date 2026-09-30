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

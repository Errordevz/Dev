import test from 'node:test';
import assert from 'node:assert/strict';
import {providerConfig} from './agent.js';

test('provider config is explicit',()=>{
  const c=providerConfig();
  assert.equal(typeof c.configured,'boolean');
  assert.equal(typeof c.url,'string');
});

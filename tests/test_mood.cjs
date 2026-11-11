const test = require('node:test');
const assert = require('node:assert/strict');
const { classifyText, classifyReadout } = require('../apps/site/assets/mood-signal.js');

test('Chinese and English positive and negative phrases', () => {
  assert.equal(classifyText('我今天很开心，也很期待明天！').state, 'good');
  assert.equal(classifyText('难过、失望，真的很疲惫。').state, 'bad');
  assert.equal(classifyText('I am happy and grateful!').score, 1);
  assert.equal(classifyText('I feel sad, lonely and anxious.').score, -1);
});

test('simple negation changes polarity without a substring happy match in unhappy', () => {
  for (const phrase of ['我不开心', '我不是很开心', '我并不觉得开心', 'I am not happy', "I don't feel happy"]) {
    assert.equal(classifyText(phrase).state, 'bad', phrase);
  }
  for (const phrase of ['我不难过', '我不太难过', '我不是不开心', 'I am not unhappy']) {
    assert.equal(classifyText(phrase).state, 'good', phrase);
  }
  assert.equal(classifyText('unhappy').evidence.length, 1);
  assert.equal(classifyText('gladness').state, 'unknown');
});

test('missing and balanced signals remain indeterminate', () => {
  for (const value of ['', '今天是星期二。', null, undefined]) {
    const result = classifyText(value);
    assert.equal(result.state, 'unknown');
    assert.equal(result.score, null);
  }
  const mixed = classifyText('开心，但也难过。');
  assert.equal(mixed.state, 'mixed');
  assert.equal(mixed.score, 0);
  assert.equal(mixed.positive, 1);
  assert.equal(mixed.negative, 1);
});

test('negation stays inside the immediate clause', () => {
  const result = classifyText('我并不难过，但有点焦虑。');
  assert.deepEqual(result.evidence.map(item => item.polarity), [1, -1]);
  assert.equal(classifyText('I am not happy, but I feel hopeful.').state, 'mixed');
});

test('readout uses exact vocabulary and global rank rather than logit sign', () => {
  const result = classifyReadout({
    zh: [{token: ' 开心', score: -8, rank: 0}],
    en: [{token: 'sad', score: -9, rank: 3}],
    other: [],
  });
  assert.equal(result.state, 'good');
  assert.equal(result.score, 0.6);
  assert.equal(classifyReadout({en: [{token: 'unhappy', score: 99, rank: 0}]}).state, 'bad');
  assert.equal(classifyReadout({en: [{token: 'Ġhappy', score: 1, rank: 0}]}).state, 'good');
  assert.equal(classifyReadout({en: [{token: 'happ', score: 1, rank: 0}]}).state, 'unknown');
});

test('malformed and nonfinite readouts cannot create a mood signal', () => {
  for (const input of [null, {}, {zh: '开心'}, {en: [null, {}, {token: 'happy', score: NaN}]}]) {
    assert.equal(classifyReadout(input).state, 'unknown');
  }
  assert.equal(classifyReadout({en: [{token: 'sad', score: 1, rank: -1}]}).score, -1);
});

test('every returned index is bounded and its evidence explains the sign', () => {
  for (let positives = 0; positives < 6; positives++) {
    for (let negatives = 0; negatives < 6; negatives++) {
      const result = classifyText('开心。'.repeat(positives) + '难过。'.repeat(negatives));
      assert.equal(result.positive, positives);
      assert.equal(result.negative, negatives);
      assert.equal(result.score, positives + negatives ? (positives - negatives) / (positives + negatives) : null);
    }
  }
});

test('OpenCode V1 idle hook forwards the matching assistant text only once', async (t) => {
  const { J7ScopeMood } = await import('../apps/serve/j7scope_serve/mood_opencode.mjs');
  const previous = process.env.J7SCOPE_MOOD_SESSION_KEY;
  const previousPort = process.env.J7SCOPE_MOOD_PORT;
  process.env.J7SCOPE_MOOD_SESSION_KEY = 'test-session';
  process.env.J7SCOPE_MOOD_PORT = '8877';
  t.after(() => {
    if (previous === undefined) delete process.env.J7SCOPE_MOOD_SESSION_KEY;
    else process.env.J7SCOPE_MOOD_SESSION_KEY = previous;
    if (previousPort === undefined) delete process.env.J7SCOPE_MOOD_PORT;
    else process.env.J7SCOPE_MOOD_PORT = previousPort;
  });
  const post = t.mock.method(global, 'fetch', async () => ({ status: 202 }));
  const plugin = await J7ScopeMood({ client: { session: { message: async ({ path }) => {
    assert.deepEqual(path, { id: 'session-a', messageID: 'reply-a' });
    return { data: { info: { role: 'assistant', id: 'reply-a' }, parts: [
      { type: 'text', text: '🙂'.repeat(510) + 'Happy' },
      { type: 'text', text: 'hidden', synthetic: true },
      { type: 'reasoning', text: 'private reasoning' },
      { type: 'tool', text: 'tool details' },
    ] } };
  } } } });
  const event = (type, properties) => plugin.event({ event: { type, properties } });
  await event('message.updated', { info: { role: 'user', id: 'user-a', sessionID: 'session-a' } });
  await event('session.idle', { sessionID: 'session-a' });
  assert.equal(post.mock.callCount(), 0);
  await event('message.updated', { info: { role: 'assistant', id: 'reply-a', sessionID: 'session-a' } });
  await event('session.idle', { sessionID: 'other-session' });
  assert.equal(post.mock.callCount(), 0);
  await event('session.idle', { sessionID: 'session-a' });
  await event('session.idle', { sessionID: 'session-a' });
  assert.equal(post.mock.callCount(), 1);
  const [url, request] = post.mock.calls[0].arguments;
  assert.equal(url, 'http://127.0.0.1:8877/mood/events');
  assert.equal(request.redirect, 'error');
  assert.equal(request.headers.Authorization, 'Bearer test-session');
  assert.deepEqual(JSON.parse(request.body), { source: 'OpenCode', phase: 'complete', text: '🙂'.repeat(495) + 'Happy' });
});

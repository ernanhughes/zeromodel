function sigmoid(x) {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

function scorePortableCritic(payload, values) {
  let logit = payload.intercept;
  for (let i = 0; i < payload.feature_ids.length; i += 1) {
    const raw = Array.isArray(values) ? values[i] : values[payload.feature_ids[i]];
    const directed = raw * payload.directionality[i];
    const z = (directed - payload.center[i]) / payload.scale[i];
    logit += payload.coefficients[i] * z;
  }
  const score = sigmoid(logit);
  let calibrated_probability = null;
  if (payload.calibration && payload.calibration.method === "platt") {
    const p = payload.calibration.parameters;
    calibrated_probability = sigmoid(p.a * logit + p.b);
  }
  return { logit, score, calibrated_probability };
}


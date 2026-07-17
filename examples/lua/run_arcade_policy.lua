-- Execute the same 112-state arcade policy exported from the ZeroModel VPM.
-- Requires only a normal Lua interpreter; no Python, NumPy, model or JSON library.

local policy_path = arg[1]
if policy_path == nil then
  error("usage: lua run_arcade_policy.lua <generated-policy.lua>")
end

local policy = dofile(policy_path)
local width = 7
local aliens = {0, 6, 1, 5}
local tank_x = math.floor(width / 2)
local cooldown = 0
local score = 0
local steps = 0
local max_steps = 32

local function state_row_id(tank, target, cooldown_value)
  local target_text = target == nil and "none" or tostring(target)
  return "tank=" .. tostring(tank)
    .. "|target=" .. target_text
    .. "|cooldown=" .. tostring(cooldown_value)
end

while #aliens > 0 and steps < max_steps do
  local target = aliens[1]
  local row_id = state_row_id(tank_x, target, cooldown)
  local action = policy.choose(row_id)
  local fired = false

  if action == "LEFT" then
    tank_x = math.max(0, tank_x - 1)
  elseif action == "RIGHT" then
    tank_x = math.min(width - 1, tank_x + 1)
  elseif action == "FIRE" then
    fired = true
    if cooldown == 0 and tank_x == target then
      table.remove(aliens, 1)
      score = score + 1
    end
  elseif action ~= "STAY" then
    error("unknown action: " .. tostring(action))
  end

  if fired and cooldown == 0 then
    cooldown = 1
  elseif not fired and cooldown > 0 then
    cooldown = cooldown - 1
  end
  steps = steps + 1
end

assert(score == 4, "expected score 4, got " .. tostring(score))
assert(#aliens == 0, "policy did not clear the wave")
assert(steps <= max_steps, "policy exceeded the step bound")

local proof = policy.read("tank=3|target=3|cooldown=0")
assert(proof.action == "FIRE")
assert(proof.artifact_id == policy.artifact_id)
assert(proof.plan_id == policy.plan_id)
assert(proof.candidates.FIRE == 1.0)

print(
  "artifact_id=" .. policy.artifact_id
    .. " plan_id=" .. policy.plan_id
    .. " score=" .. tostring(score)
    .. " cleared=true"
    .. " steps=" .. tostring(steps)
)

# Failure Detection Threshold - Updated Logic

## Problem with 0.15%
- 0.15% of 26100 = **39 points**
- Today's stop losses: 16-76 points
- **Would trigger almost immediately!** ❌

---

## New Logic: % of Distance to Stop Loss

Instead of absolute percentage, we now use **how close you are to stop loss**:

### Example 1: Today's 09:46 SHORT Signal
```
Entry: 26123
Stop Loss: 26199
SL Distance: 76 points

At 26150 (27 pts against us):
  35% to SL → ✅ WARNING alert sent

At 26176 (53 pts against us):
  70% to SL → 🚨 URGENT alert sent
```

### Example 2: Tight Stop Loss Signal
```
Entry: 26181
Stop Loss: 26197  
SL Distance: 16 points

At 26186 (5 pts against us):
  31% to SL → ✅ WARNING alert sent

At 26192 (11 pts against us):
  69% to SL → 🚨 URGENT alert sent
```

---

## Trigger Levels

**WARNING** (30% to SL):
- Early heads-up
- Consider tightening stop
- Monitor closely

**URGENT** (70% to SL):
- Very close to stop loss
- Likely to hit SL soon
- Consider exiting

**Max 2 alerts per signal** (prevents spam)

---

## Practical Impact

### Tight SL (20 points):
- WARNING at ~6 points against (30%)
- URGENT at ~14 points against (70%)

### Normal SL (50 points):
- WARNING at ~15 points against (30%)
- URGENT at ~35 points against (70%)

### Wide SL (75 points):
- WARNING at ~23 points against (30%)
- URGENT at ~53 points against (70%)

---

## Alert Format

```
⚠️ **SIGNAL FAILING** 📉

📊 **NIFTY**
Signal: SHORT @ 26123
Stop Loss: 26199

💰 Current: 26150
📉 Moved: 27 points against (35% to SL)

⚡ WARNING: Consider tightening stop or exiting
```

vs

```
🚨 **SIGNAL FAILING URGENTLY** 📉

📊 **NIFTY**  
Signal: SHORT @ 26123
Stop Loss: 26199

💰 Current: 26176
📉 Moved: 53 points against (70% to SL)

🚨 URGENT: Very close to stop loss!
```

---

**Much more practical!** 🎯

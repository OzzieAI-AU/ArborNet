# NedK perfected patch (2026-09-29)

## Critical fixes
1. **NedKPrimeBlock2D.Parameters()** — implemented; yields norm/attn/FFN/loopNorm params in stable order.
2. **NedKOzComplete.Materialize** — tensor length vs **element count** (not Rank).
3. **Defaults R=4** — DefaultLoops / Infer Loops / Forward() default use MaxLoops / Config.
4. **NedKConfig.WinnerArchitectureId** — `dense-ffn-parallel-deltanorm` (matches Block2D).
5. **Program.Evaluate** — default loops = `DefaultInferenceLoops`; maxValRise from Config.

## Copy into ArborNet/NedK/
Replace the matching files, then:
```
dotnet build NedK.csproj -c Release
dotnet run --project NedK.csproj -c Release
```

## Do not claim
- Micro residual-loop train loss as LM CE.
- KD 95 phantom baseline as gate.

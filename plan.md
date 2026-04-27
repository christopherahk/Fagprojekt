## Plan: Detailed Port To USB Online Loop

Replace one-shot epoch training with a frame-driven runtime: Arduino continuously parses serial frames into a 56x100 tensor, always runs inference, returns binary signal/no-signal + confidences, and performs one training update only when a 3-class one-hot label is attached.

**Steps**
1. Phase 1 - Arduino Runtime Refactor in Fagprojekt.ino
2. Remove demo-only dataset helpers and epoch loop entry points:
3. Remove makeInputTensor() at Fagprojekt.ino:143.
4. Remove makeDummyOneHotLabels() at Fagprojekt.ino:156.
5. Remove accuracyFromOneHot() at Fagprojekt.ino:183. CHECK
6. Replace runTraining() at Fagprojekt.ino:197 with a frame processor function, e.g. processFrame(Tensor& X, const Tensor* labelOrNull). CHECK
7. Replace one-shot loop control at Fagprojekt.ino:268 with continuous polling logic that reads from Serial whenever bytes are available.
8. Keep model constants and DenseLayer implementation (Fagprojekt.ino:7-125), but instantiate model objects once (not per frame) so weights persist during runtime.

9. Phase 2 - Add Serial Parsing State Machine in Fagprojekt.ino (*depends on Phase 1*)
10. Add protocol parser state variables at file scope: current channel index [0..55], line buffer, frame-ready flag, optional label-present flag, error counter.
11. Add reusable input tensor buffer Tensor frameX(INPUT_ROWS, INPUT_COLS) and label buffer Tensor frameY(1, OUTPUT_CLASSES).
12. Implement readSerialLineNonBlocking(): append chars until newline; ignore CR; enforce max line length; return complete line when available.
13. Implement parseDataLineIntoChannel(line, frameX, channelIdx): split by comma; expect exactly INPUT_COLS values; store at frameX(channelIdx, col); increment channelIdx on success.
14. Implement parseLabelLine(line, frameY): parse 3 floats; validate one-hot (sum close to 1, each in {0,1} with tolerance).
15. Define frame completion rule with current simple protocol:
16. Data frame is complete when 56 valid channel lines are received.
17. Label line is optional and can appear after 56th channel line before next frame starts.
18. Metadata lines beginning with #META should be ignored (or parsed later) without breaking frame assembly.
19. On malformed lines: drop partial frame, reset parser state, emit ERR_FRAME_RESET message.

20. Phase 3 - Inference Output Path in Fagprojekt.ino (*depends on Phase 2*)
21. In processFrame(), run existing forward path:
22. dense1.forward(frameX), activation1.forward(dense1.output, 0.01f), dense2.forward(activation1.output).
23. Compute probabilities via lossActivation.activation.forward(dense2.output) for inference-only flow.
24. Derive class index from row 0 argmax.
25. Derive binary flag per your decision: class 0 -> no signal (0), class 1 or 2 -> signal (1).
26. Emit structured serial response line, e.g. OUT,pred=2,signal=1,p0=...,p1=...,p2=... .

27. Phase 4 - Conditional Online Training in Fagprojekt.ino (*depends on Phase 3*)
28. If label-present for this frame:
29. Compute loss with lossActivation.forward(dense2.output, frameY).
30. Guard: if loss NaN/Inf, skip update and print ERR_NUMERIC.
31. Backprop in existing reverse order: lossActivation.backward(frameY), dense2.backward(...), activation1.backward(...), dense1.backward(..., false).
32. Apply updates: dense2.update(LEARNING_RATE), dense1.update(LEARNING_RATE).
33. Emit TRAIN_OK with loss and true class.
34. If label not present: skip backward/update and emit INFER_ONLY.
35. Add lightweight counters and periodic status print every N frames: rx_ok, rx_drop, infer_count, train_count, numeric_err.

36. Phase 5 - Python Stream Contract Extensions in python_files/streamingrhd.py (*parallel with Phases 2-4 once line format is fixed*)
37. Keep existing payload generation unchanged in stream_rhd_file() around lines 261-283 (56 channel lines of 100 values each).
38. Extend write_payload() at python_files/streamingrhd.py:99 to optionally append one label line after payload_batch for supervised chunks.
39. Add label provider hook in stream_rhd_file() (near chunk_idx loop at line 237): determine optional class label for current chunk and convert to 3-long one-hot text line.
40. Keep SEND_METADATA behavior intact (python_files/streamingrhd.py:31 and function at line 83), and ensure Arduino ignores #META safely.
41. Add optional serial response reader loop after write_payload() call (python_files/streamingrhd.py:300) to print/log OUT / TRAIN_OK / ERR_* responses.

42. Phase 6 - Setup/Loop Lifecycle Cleanup in Fagprojekt.ino (*depends on Phases 2-4*)
43. setup() at Fagprojekt.ino:266 should initialize Serial and model weights once.
44. loop() at Fagprojekt.ino:268 should only:
45. pump serial parser,
46. call processFrame() when frame-ready,
47. reset frame state,
48. call yield() periodically.
49. Ensure there is no TRAIN_EPOCHS dependency anywhere.

50. Phase 7 - Verification Checklist
51. Compile sketch after removing runTraining()/TRAIN_EPOCHS path.
52. Arduino manual serial test: feed 56 valid lines without label -> OUT + INFER_ONLY.
53. Feed 56 valid lines + valid one-hot label -> OUT + TRAIN_OK + finite loss.
54. Feed malformed channel length (<100 or >100 floats) -> ERR_FRAME_RESET and successful next-frame recovery.
55. Run python_files/streamingrhd.py in live_online for >=50 frames and verify stable parser counters.
56. Confirm binary mapping correctness by sending controlled logits/known inputs where class 0 and class 1/2 are expected.

**Relevant files**
- vsls:/Fagprojekt.ino — primary refactor target.
- vsls:/python_files/streamingrhd.py — attach optional labels and consume Arduino replies.
- vsls:/Tensor.cpp — monitor allocation behavior in operator= at Tensor.cpp:118 if parser design introduces frequent reassignment (avoid by writing directly into preallocated Tensor buffers).

**Decisions**
- Included: continuous runtime loop; immediate per-frame training when label exists.
- Included: 3-class model retained; binary output derived by mapping class0=0 and class1/2=1.
- Included: simple line-based serial protocol retained.
- Included: RAM-only weight persistence.
- Excluded: checksum/ACK framing protocol redesign.
- Excluded: flash/EEPROM model checkpoints.

**Further Considerations**
1. Recommended minimal protocol marker even with line-based mode: add one FRAME_DONE token from Python after 56 lines so Arduino can disambiguate delayed label arrival.
2. If throughput is unstable on Nano 33 BLE Sense Rev2, first mitigation should be lowering STREAM_RATE_HZ in python_files/streamingrhd.py.
3. If supervised labels are unavailable for many frames, preserve simple inference-only operation without stalling parser waiting for labels.

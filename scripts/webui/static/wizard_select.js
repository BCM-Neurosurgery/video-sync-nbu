(() => {
  "use strict";

  const SELECTORS = {
    modeSelect: "#selection_mode",
    continueButton: "#continue_btn",
    selectAllButton: "#select_all_pairs",
    segmentsSection: "#segments_section",
    timeSection: "#time_section",
    sampleSection: "#sample_section",
    targetPairs: "input[name='target_pairs']",
    rangeWidgets: ".range-widget",
    cameraRows: "tr[data-mode='time-camera'], tr[data-mode='sample-camera']",
    timeRows: "tr[data-mode='time-camera']",
    sampleRows: "tr[data-mode='sample-camera']",
    cameraEnabledInput: "input[data-role='camera-enabled']",
    timeStartInput: "input[data-role='time-start']",
    timeEndInput: "input[data-role='time-end']",
    sampleStartInput: "input[data-role='sample-start']",
    sampleEndInput: "input[data-role='sample-end']",
    timeZoneInput: "select[data-role='time-zone']",
    rangeTrack: ".range-track",
    rangeStartHandle: "[data-role='range-start']",
    rangeEndHandle: "[data-role='range-end']",
    rangeFill: "[data-role='range-fill']",
    rangeStartLabel: "[data-role='range-start-label']",
    rangeEndLabel: "[data-role='range-end-label']",
    matrixToggle: ".matrix-toggle",
    cameraToggle: ".camera-toggle",
  };

  const TimeUtils = {
    normalizeTimeText(value) {
      if (!value) {
        return "";
      }
      return String(value).trim().replace("T", " ").slice(0, 19);
    },

    asInt(value) {
      if (value === null || value === undefined || String(value).trim() === "") {
        return null;
      }
      const n = Number(value);
      return Number.isFinite(n) ? Math.trunc(n) : null;
    },

    zoneDatePartsFromEpoch(seconds, timeZone) {
      try {
        const dtf = new Intl.DateTimeFormat("en-US", {
          timeZone: timeZone || "UTC",
          year: "numeric",
          month: "2-digit",
          day: "2-digit",
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        });
        const parts = dtf.formatToParts(new Date(Math.trunc(seconds) * 1000));
        const out = {};
        for (const part of parts) {
          if (part.type !== "literal") {
            out[part.type] = part.value;
          }
        }
        return {
          year: Number(out.year),
          month: Number(out.month),
          day: Number(out.day),
          hour: Number(out.hour),
          minute: Number(out.minute),
          second: Number(out.second),
        };
      } catch (_) {
        return null;
      }
    },

    zoneOffsetSecondsAtEpoch(seconds, timeZone) {
      const parts = this.zoneDatePartsFromEpoch(seconds, timeZone);
      if (!parts) {
        return 0;
      }
      const asUtc = Math.trunc(
        Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second) /
          1000
      );
      return asUtc - Math.trunc(seconds);
    },

    parseDateTimeToUtcSecondsInZone(raw, timeZone) {
      const text = this.normalizeTimeText(raw);
      const match = /^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$/.exec(text);
      if (!match) {
        return null;
      }

      const year = Number(match[1]);
      const month = Number(match[2]);
      const day = Number(match[3]);
      const hour = Number(match[4]);
      const minute = Number(match[5]);
      const second = Number(match[6]);

      if ([year, month, day, hour, minute, second].some((n) => !Number.isFinite(n))) {
        return null;
      }

      if (!timeZone || timeZone === "UTC") {
        return Math.trunc(Date.UTC(year, month - 1, day, hour, minute, second) / 1000);
      }

      const localAsUtc = Math.trunc(Date.UTC(year, month - 1, day, hour, minute, second) / 1000);
      let guess = localAsUtc;
      for (let i = 0; i < 5; i += 1) {
        const offset = this.zoneOffsetSecondsAtEpoch(guess, timeZone);
        const next = localAsUtc - offset;
        if (Math.abs(next - guess) < 1) {
          guess = next;
          break;
        }
        guess = next;
      }
      return Math.trunc(guess);
    },

    formatUtcSecondsToDateTime(seconds) {
      if (!Number.isFinite(seconds)) {
        return "";
      }
      const dt = new Date(Math.trunc(seconds) * 1000);
      const year = dt.getUTCFullYear();
      const month = String(dt.getUTCMonth() + 1).padStart(2, "0");
      const day = String(dt.getUTCDate()).padStart(2, "0");
      const hour = String(dt.getUTCHours()).padStart(2, "0");
      const minute = String(dt.getUTCMinutes()).padStart(2, "0");
      const second = String(dt.getUTCSeconds()).padStart(2, "0");
      return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
    },

    formatUtcSecondsToDateTimeInZone(seconds, timeZone) {
      if (!Number.isFinite(seconds)) {
        return "";
      }
      if (!timeZone || timeZone === "UTC") {
        return this.formatUtcSecondsToDateTime(seconds);
      }

      const parts = this.zoneDatePartsFromEpoch(seconds, timeZone);
      if (!parts) {
        return this.formatUtcSecondsToDateTime(seconds);
      }

      const year = String(parts.year).padStart(4, "0");
      const month = String(parts.month).padStart(2, "0");
      const day = String(parts.day).padStart(2, "0");
      const hour = String(parts.hour).padStart(2, "0");
      const minute = String(parts.minute).padStart(2, "0");
      const second = String(parts.second).padStart(2, "0");
      return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
    },
  };

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function createRangeWidgetController(widget, { onChange }) {
    const row = widget.closest("tr");
    if (!row) {
      return;
    }

    const kind = (widget.dataset.rangeKind || "").trim();
    const track = widget.querySelector(SELECTORS.rangeTrack);
    const startHandle = widget.querySelector(SELECTORS.rangeStartHandle);
    const endHandle = widget.querySelector(SELECTORS.rangeEndHandle);
    const fill = widget.querySelector(SELECTORS.rangeFill);
    if (!track || !startHandle || !endHandle || !fill) {
      return;
    }

    const rawAllowedMin = (widget.dataset.rangeMin || "").trim();
    const rawAllowedMax = (widget.dataset.rangeMax || "").trim();
    const rawStart = (widget.dataset.rangeStart || "").trim();
    const rawEnd = (widget.dataset.rangeEnd || "").trim();
    const boundSourceTimeZone = (widget.dataset.rangeBoundTz || "UTC").trim() || "UTC";
    const valueSourceTimeZone = (widget.dataset.rangeValueTz || "UTC").trim() || "UTC";
    const rawScaleMin = (widget.dataset.rangeScaleMin || "").trim();
    const rawScaleMax = (widget.dataset.rangeScaleMax || "").trim();
    const scaleSourceTimeZone =
      (widget.dataset.rangeScaleTz || boundSourceTimeZone).trim() || boundSourceTimeZone;

    const timeZoneSelect = row.querySelector(SELECTORS.timeZoneInput);
    const activeTimeZone = () => {
      if (kind !== "time") {
        return "UTC";
      }
      return (timeZoneSelect?.value || "UTC").trim() || "UTC";
    };

    const parseRaw = (value, timeZone) => {
      if (kind === "time") {
        return TimeUtils.parseDateTimeToUtcSecondsInZone(value, timeZone);
      }
      return TimeUtils.asInt(value);
    };

    const formatRaw = (value) => {
      if (kind === "time") {
        return TimeUtils.formatUtcSecondsToDateTimeInZone(value, activeTimeZone());
      }
      return String(Math.trunc(value));
    };

    let allowedMin = parseRaw(rawAllowedMin, boundSourceTimeZone);
    let allowedMax = parseRaw(rawAllowedMax, boundSourceTimeZone);
    let startValue = parseRaw(rawStart, valueSourceTimeZone);
    let endValue = parseRaw(rawEnd, valueSourceTimeZone);
    let scaleMin = parseRaw(rawScaleMin, scaleSourceTimeZone);
    let scaleMax = parseRaw(rawScaleMax, scaleSourceTimeZone);

    if (allowedMin === null) {
      allowedMin = startValue;
    }
    if (allowedMax === null) {
      allowedMax = endValue;
    }
    if (startValue === null) {
      startValue = allowedMin;
    }
    if (endValue === null) {
      endValue = allowedMax;
    }
    if (scaleMin === null) {
      scaleMin = allowedMin;
    }
    if (scaleMax === null) {
      scaleMax = allowedMax;
    }

    if (
      [allowedMin, allowedMax, scaleMin, scaleMax, startValue, endValue].some((v) => v === null)
    ) {
      startHandle.setAttribute("disabled", "disabled");
      endHandle.setAttribute("disabled", "disabled");
      fill.style.left = "0%";
      fill.style.width = "0%";
      return;
    }

    if (allowedMax < allowedMin) {
      const temp = allowedMin;
      allowedMin = allowedMax;
      allowedMax = temp;
    }
    if (scaleMax < scaleMin) {
      const temp = scaleMin;
      scaleMin = scaleMax;
      scaleMax = temp;
    }

    scaleMin = Math.min(scaleMin, allowedMin);
    scaleMax = Math.max(scaleMax, allowedMax);

    startValue = clamp(startValue, allowedMin, allowedMax);
    endValue = clamp(endValue, allowedMin, allowedMax);
    if (startValue > endValue) {
      startValue = endValue;
    }
    if (startValue === endValue && allowedMax > allowedMin) {
      startValue = allowedMin;
      endValue = allowedMax;
    }

    const span = Math.max(scaleMax - scaleMin, 1);
    const valueToPct = (value) => ((value - scaleMin) / span) * 100;
    const pctToValue = (pct) => scaleMin + Math.round((clamp(pct, 0, 100) / 100) * span);

    const startField = row.querySelector(
      kind === "time" ? SELECTORS.timeStartInput : SELECTORS.sampleStartInput
    );
    const endField = row.querySelector(
      kind === "time" ? SELECTORS.timeEndInput : SELECTORS.sampleEndInput
    );
    const startLabel = row.querySelector(SELECTORS.rangeStartLabel);
    const endLabel = row.querySelector(SELECTORS.rangeEndLabel);

    const state = {
      start: startValue,
      end: endValue,
    };

    const parseEditedValue = (rawText) => {
      if (kind === "time") {
        return TimeUtils.parseDateTimeToUtcSecondsInZone(rawText, activeTimeZone());
      }
      return TimeUtils.asInt(rawText);
    };

    const render = () => {
      const startRounded = Math.round(state.start);
      const endRounded = Math.round(state.end);
      const startText = formatRaw(startRounded);
      const endText = formatRaw(endRounded);

      if (startField) {
        startField.value = startText;
      }
      if (endField) {
        endField.value = endText;
      }
      if (startLabel) {
        startLabel.textContent = startText;
      }
      if (endLabel) {
        endLabel.textContent = endText;
      }

      const left = valueToPct(startRounded);
      const right = valueToPct(endRounded);
      startHandle.style.left = `${left}%`;
      endHandle.style.left = `${right}%`;
      fill.style.left = `${left}%`;
      fill.style.width = `${Math.max(right - left, 0)}%`;
      onChange();
    };

    const applyEditedValue = (which, rawText) => {
      const parsed = parseEditedValue(rawText);
      if (parsed === null) {
        return false;
      }
      const value = clamp(parsed, allowedMin, allowedMax);
      if (which === "start") {
        state.start = Math.min(value, state.end);
      } else {
        state.end = Math.max(value, state.start);
      }
      render();
      return true;
    };

    const clientXToValue = (clientX) => {
      const rect = track.getBoundingClientRect();
      if (!rect.width) {
        return state.start;
      }
      const pct = ((clientX - rect.left) / rect.width) * 100;
      return pctToValue(pct);
    };

    const startDrag = (which, clientX) => {
      if (startHandle.disabled || endHandle.disabled) {
        return;
      }

      const move = (ev) => {
        const value = clamp(clientXToValue(ev.clientX), allowedMin, allowedMax);
        if (which === "start") {
          state.start = Math.min(value, state.end);
        } else {
          state.end = Math.max(value, state.start);
        }
        render();
      };

      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };

      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up, { once: true });
      move({ clientX });
    };

    const openInlineEditor = (which) => {
      if (startHandle.disabled || endHandle.disabled) {
        return;
      }

      const label = which === "start" ? startLabel : endLabel;
      const field = which === "start" ? startField : endField;
      if (!label || !field || label.dataset.editing === "1") {
        return;
      }

      label.dataset.editing = "1";
      label.style.display = "none";

      const input = document.createElement("input");
      input.type = "text";
      input.className = "range-edit-input mono";
      input.value = String(field.value || "").trim();
      label.insertAdjacentElement("afterend", input);
      input.focus();
      input.select();

      let finished = false;
      const closeEditor = (apply) => {
        if (finished) {
          return;
        }
        finished = true;

        if (apply) {
          const ok = applyEditedValue(which, input.value);
          if (!ok) {
            render();
          }
        }

        input.remove();
        label.style.display = "";
        label.dataset.editing = "0";
      };

      input.addEventListener("keydown", (ev) => {
        if (ev.key === "Enter") {
          ev.preventDefault();
          closeEditor(true);
        } else if (ev.key === "Escape") {
          ev.preventDefault();
          closeEditor(false);
        }
      });
      input.addEventListener("blur", () => closeEditor(true));
    };

    startHandle.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      startDrag("start", ev.clientX);
    });

    endHandle.addEventListener("pointerdown", (ev) => {
      ev.preventDefault();
      startDrag("end", ev.clientX);
    });

    track.addEventListener("pointerdown", (ev) => {
      if (ev.target === startHandle || ev.target === endHandle) {
        return;
      }
      const value = clientXToValue(ev.clientX);
      const distStart = Math.abs(value - state.start);
      const distEnd = Math.abs(value - state.end);
      startDrag(distStart <= distEnd ? "start" : "end", ev.clientX);
    });

    if (startLabel) {
      startLabel.addEventListener("dblclick", () => openInlineEditor("start"));
    }
    if (endLabel) {
      endLabel.addEventListener("dblclick", () => openInlineEditor("end"));
    }

    if (kind === "time" && timeZoneSelect) {
      timeZoneSelect.addEventListener("change", () => {
        widget.dataset.rangeValueTz = activeTimeZone();
        render();
      });
    }

    render();
  }

  function createSelectionController(doc) {
    const modeSelect = doc.querySelector(SELECTORS.modeSelect);
    const continueButton = doc.querySelector(SELECTORS.continueButton);
    const selectAllButton = doc.querySelector(SELECTORS.selectAllButton);

    const pairInputs = () => Array.from(doc.querySelectorAll(SELECTORS.targetPairs));
    const userEnabledInputs = (inputs) =>
      inputs.filter((input) => !input.disabled && input.dataset.baseDisabled !== "1");

    const activeMode = () => (modeSelect?.value || "segments").trim();

    const isCameraRowEnabled = (row) => {
      const marker = row.querySelector(SELECTORS.cameraEnabledInput);
      return !marker || marker.value !== "0";
    };

    const setCameraRowEnabled = (row, enabled) => {
      const marker = row.querySelector(SELECTORS.cameraEnabledInput);
      if (marker) {
        marker.value = enabled ? "1" : "0";
      }
      row.classList.toggle("camera-disabled", !enabled);
      row.querySelectorAll("input, select, button.range-thumb").forEach((el) => {
        if (el.dataset.role === "camera-enabled") {
          return;
        }
        el.disabled = !enabled;
      });
    };

    const setSectionVisibility = () => {
      const mode = activeMode();
      const segments = doc.querySelector(SELECTORS.segmentsSection);
      const time = doc.querySelector(SELECTORS.timeSection);
      const sample = doc.querySelector(SELECTORS.sampleSection);
      if (segments) {
        segments.style.display = mode === "segments" ? "block" : "none";
      }
      if (time) {
        time.style.display = mode === "time" ? "block" : "none";
      }
      if (sample) {
        sample.style.display = mode === "sample" ? "block" : "none";
      }
    };

    const validateTimeTable = () => {
      const rows = Array.from(doc.querySelectorAll(SELECTORS.timeRows));
      if (!rows.length) {
        return false;
      }
      const activeRows = rows.filter((row) => isCameraRowEnabled(row));
      if (!activeRows.length) {
        return false;
      }
      return activeRows.every((row) => {
        const start = TimeUtils.normalizeTimeText(
          row.querySelector(SELECTORS.timeStartInput)?.value || ""
        );
        const end = TimeUtils.normalizeTimeText(
          row.querySelector(SELECTORS.timeEndInput)?.value || ""
        );
        return !!start && !!end;
      });
    };

    const validateSampleTable = () => {
      const rows = Array.from(doc.querySelectorAll(SELECTORS.sampleRows));
      if (!rows.length) {
        return false;
      }
      const activeRows = rows.filter((row) => isCameraRowEnabled(row));
      if (!activeRows.length) {
        return false;
      }
      return activeRows.every((row) => {
        const start = TimeUtils.asInt(row.querySelector(SELECTORS.sampleStartInput)?.value || "");
        const end = TimeUtils.asInt(row.querySelector(SELECTORS.sampleEndInput)?.value || "");
        return start !== null && end !== null && end >= start;
      });
    };

    const updateContinueState = () => {
      if (!continueButton) {
        return;
      }
      const mode = activeMode();
      if (mode === "segments") {
        const anyChecked = pairInputs().some(
          (input) => input.checked && input.dataset.baseDisabled !== "1"
        );
        continueButton.disabled = !anyChecked;
        return;
      }
      if (mode === "time") {
        continueButton.disabled = !validateTimeTable();
        return;
      }
      continueButton.disabled = !validateSampleTable();
    };

    const updateSelectAllLabel = () => {
      if (!selectAllButton) {
        return;
      }
      const inputs = userEnabledInputs(pairInputs());
      const allChecked = inputs.length > 0 && inputs.every((input) => input.checked);
      selectAllButton.textContent = allChecked ? "Select none" : "Select all";
    };

    const toggleSet = (inputs) => {
      const enabled = userEnabledInputs(inputs);
      if (!enabled.length) {
        return;
      }
      const allChecked = enabled.every((input) => input.checked);
      enabled.forEach((input) => {
        input.checked = !allChecked;
      });
    };

    const bindEvents = () => {
      modeSelect?.addEventListener("change", () => {
        setSectionVisibility();
        updateContinueState();
      });

      if (selectAllButton) {
        selectAllButton.addEventListener("click", () => {
          const inputs = userEnabledInputs(pairInputs());
          const allChecked = inputs.length > 0 && inputs.every((input) => input.checked);
          inputs.forEach((input) => {
            input.checked = !allChecked;
          });
          updateSelectAllLabel();
          updateContinueState();
        });
      }

      doc.addEventListener("click", (event) => {
        const target = event.target;
        if (!target) {
          return;
        }

        const cameraToggle = target.closest(SELECTORS.cameraToggle);
        if (cameraToggle) {
          const row = cameraToggle.closest(SELECTORS.cameraRows);
          if (row) {
            setCameraRowEnabled(row, !isCameraRowEnabled(row));
            updateContinueState();
          }
          return;
        }

        const matrixToggle = target.closest(SELECTORS.matrixToggle);
        if (!matrixToggle) {
          return;
        }

        const segment = matrixToggle.getAttribute("data-seg");
        const camera = matrixToggle.getAttribute("data-cam");
        if (segment) {
          const inputs = userEnabledInputs(pairInputs()).filter(
            (input) => input.getAttribute("data-seg") === segment
          );
          toggleSet(inputs);
          updateContinueState();
        } else if (camera) {
          const inputs = userEnabledInputs(pairInputs()).filter(
            (input) => input.getAttribute("data-cam") === camera
          );
          toggleSet(inputs);
          updateSelectAllLabel();
          updateContinueState();
        }
      });

      doc.addEventListener("change", () => {
        updateSelectAllLabel();
        updateContinueState();
      });

      doc.addEventListener("input", (event) => {
        const target = event.target;
        if (!target) {
          return;
        }
        if (
          target.matches(SELECTORS.timeZoneInput) ||
          target.matches(SELECTORS.sampleStartInput) ||
          target.matches(SELECTORS.sampleEndInput)
        ) {
          updateContinueState();
        }
      });
    };

    const init = () => {
      doc.querySelectorAll(SELECTORS.rangeWidgets).forEach((widget) => {
        createRangeWidgetController(widget, { onChange: updateContinueState });
      });

      doc.querySelectorAll(SELECTORS.cameraRows).forEach((row) => {
        setCameraRowEnabled(row, isCameraRowEnabled(row));
      });

      bindEvents();
      setSectionVisibility();
      updateSelectAllLabel();
      updateContinueState();
    };

    return {
      init,
    };
  }

  createSelectionController(document).init();
})();

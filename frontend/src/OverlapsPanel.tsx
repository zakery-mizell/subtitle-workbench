import type { RestoreEngineName } from "./lib/restore";
import type { OverlapRegion } from "./types";

interface OverlapsPanelProps {
  regions: OverlapRegion[];
  restoreSoloTracks: boolean;
  onRestoreSoloTracksChange: (value: boolean) => void;
  restoreEngine: RestoreEngineName;
  onRestoreEngineChange: (value: RestoreEngineName) => void;
}

/**
 * Settings that back the overlap tooling. The overlap regions themselves live
 * on the main waveform — highlighted bands that open a dock of controls when
 * clicked — so this tab only carries the global knobs.
 */
export function OverlapsPanel({
  regions,
  restoreSoloTracks,
  onRestoreSoloTracksChange,
  restoreEngine,
  onRestoreEngineChange,
}: OverlapsPanelProps) {
  return (
    <section className="selection-panel mastering-panel overlaps-panel">
      <div className="panel-section-heading">
        <p className="eyebrow">Overlaps</p>
        <h3>Untangle simultaneous speech</h3>
      </div>
      <p className="helper-text">
        AI separation (UniSE) for the moments where speakers talk over each other. Overlaps are highlighted directly
        on the main waveform — scroll on it to zoom in, click a band to spotlight or mute a voice, and process from
        the controls that appear underneath.
      </p>

      {!regions.length ? (
        <p className="helper-text">
          No overlapping speech was detected in this recording. Overlaps are found during transcription when more
          than one speaker is configured, so retranscribe with the right speaker count if something is missing.
        </p>
      ) : null}

      {regions.length ? (
        <>
          <div className="panel-section-heading">
            <p className="eyebrow">Per-speaker muting</p>
            <h3>Per-speaker solo tracks</h3>
          </div>
          <p className="helper-text">
            The transport bar&apos;s speaker toggles mute one voice at a time, even inside overlaps.
            These tracks are prepared automatically.
          </p>
          <p className="helper-text">
            UniSE runs automatically only around detected overlaps and rapid speaker handoffs. Those short windows
            are checked voice by voice; everywhere else keeps the untouched original sound. Each assembled speaker
            track is also transcribed for per-speaker subtitle information. Short pause-bounded replies are additionally
            compared with each speaker&apos;s original voice signature when diarization may have missed the handoff entirely.
          </p>
          <label className="mastering-toggle">
            <input
              type="checkbox"
              checked={restoreSoloTracks}
              onChange={(event) => onRestoreSoloTracksChange(event.target.checked)}
            />
            <span>Restore voices</span>
          </label>
          {restoreSoloTracks ? (
            <label>
              Engine
              <select
                value={restoreEngine}
                onChange={(event) => onRestoreEngineChange(event.target.value as RestoreEngineName)}
              >
                <option value="sidon">Sidon — multilingual, 48 kHz</option>
                <option value="diamond">Diamond — English, 44.1 kHz</option>
              </select>
            </label>
          ) : null}
          <p className="helper-text">
            {restoreEngine === "sidon"
              ? "Cleans each solo track after isolation and returns 48 kHz. Multilingual, and about 4× faster than real-time on this Mac."
              : "Regenerates each solo track as 44.1 kHz speech after isolation. English-trained and roughly 10× slower than real-time on this Mac."}{" "}
            Changing either setting re-renders the solo tracks.
          </p>
        </>
      ) : null}
    </section>
  );
}

export default OverlapsPanel;

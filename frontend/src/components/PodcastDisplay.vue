<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
  conversationHistory: {
    type: Array,
    default: () => []
  },
  // NEW: Prop to receive the audio log dictionary
  audioLog: {
    type: Object,
    default: () => ({})
  }
});

const audioPlayer = ref(null);
const BACKEND_BASE_URL = 'http://127.0.0.1:8000';

const currentlyPlayingFile = ref(null);
const isPlaying = ref(false);

const formattedHistory = computed(() => {
  if (!props.conversationHistory || props.conversationHistory.length === 0) {
    return [];
  }
  // REWRITTEN: This logic now correctly combines the clean history with the audio log.
  return props.conversationHistory.map((cleanEntry, index) => {
    // Get the corresponding audio file from the log using the turn's index
    const audioFile = props.audioLog[index] || null;

    let speaker = "System";
    let message = cleanEntry;
    if (cleanEntry.startsWith("Initial Topic:")) {
      speaker = "Topic";
      message = cleanEntry.replace("Initial Topic: ", "");
    } else if (cleanEntry.startsWith("Curious Casey:")) {
      speaker = "Curious Casey";
      message = cleanEntry.replace("Curious Casey: ", "");
    } else if (cleanEntry.startsWith("Factual Finn (addressing doubt):")) {
      speaker = "Factual Finn";
      message = cleanEntry.replace("Factual Finn (addressing doubt): ", "");
    } else if (cleanEntry.startsWith("Factual Finn:")) {
      speaker = "Factual Finn";
      message = cleanEntry.replace("Factual Finn: ", "");
    } else if (cleanEntry.startsWith("User Doubt:")) {
      speaker = "Your Question";
      message = cleanEntry.replace("User Doubt: ", "");
    }
    // No more string parsing for audio files!
    return { speaker, message, audioFile };
  });
});

// Play/pause logic is correct and remains the same.
const playSegment = (filename) => {
    if (!audioPlayer.value || !filename) return;

    const audioUrl = `${BACKEND_BASE_URL}/get_podcast_audio/${filename}`;

    if (currentlyPlayingFile.value === filename && isPlaying.value) {
        audioPlayer.value.pause();
    } else {
        currentlyPlayingFile.value = filename;
        audioPlayer.value.src = audioUrl;
        audioPlayer.value.play();
    }
};

const handleAudioPlay = () => {
    isPlaying.value = true;
};
const handleAudioPauseOrEnd = () => {
    isPlaying.value = false;
};
</script>

<template>
  <div class="podcast-display" v-if="formattedHistory.length > 0">
    <audio 
      ref="audioPlayer" 
      style="display: none;"
      @play="handleAudioPlay"
      @pause="handleAudioPauseOrEnd"
      @ended="handleAudioPauseOrEnd"
    ></audio>
    <h2>Podcast Conversation</h2>
    <div class="conversation-log">
      <div v-for="(turn, index) in formattedHistory" :key="index" :class="['turn', turn.speaker.toLowerCase().replace(/\s+/g, '-')]">
        <div class="turn-header">
            <strong class="speaker">{{ turn.speaker }}:</strong>
            <button 
              v-if="turn.audioFile" 
              @click="playSegment(turn.audioFile)" 
              class="play-btn" 
              :title="isPlaying && currentlyPlayingFile === turn.audioFile ? 'Pause line' : 'Play line'"
            >
                {{ isPlaying && currentlyPlayingFile === turn.audioFile ? '❚❚' : '▶' }}
            </button>
        </div>
        <p class="message">{{ turn.message }}</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.podcast-display {
  margin-top: 2rem;
  padding: 25px;
  background-color: #2f2f2f;
  border-radius: 8px;
  text-align: left;
}
.conversation-log .turn {
  margin-bottom: 1.25rem;
  padding-bottom: 1.25rem;
  border-bottom: 1px solid #444;
}
.conversation-log .turn:last-child {
  border-bottom: none;
  margin-bottom: 0;
  padding-bottom: 0;
}
.turn-header {
    display: flex;
    align-items: center;
    margin-bottom: 5px;
}
.speaker {
  font-weight: bold;
  margin-right: 8px;
}
.play-btn {
    background: #555;
    color: #fff;
    border: none;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    font-size: 12px;
    line-height: 24px;
    text-align: center;
    cursor: pointer;
    padding: 0;
    /* Optical alignment for play/pause icons */
    padding-left: 1px;
    display: flex;
    justify-content: center;
    align-items: center;
    transition: background-color 0.2s;
}
.play-btn:hover {
    background: #007bff;
}
.play-btn:active {
    background: #0056b3;
}

.turn.curious-casey .speaker, .turn.narrator .speaker {
  color: #63a4ff;
}
.turn.factual-finn .speaker, .turn.factual-finn-(addressing-doubt) .speaker {
  color: #ff9900;
}

.turn.your-question .speaker {
  color: #33cc33;
}
.turn.topic .speaker {
  color: #ccc;
}
.message {
  margin-top: 5px;
  margin-bottom: 0;
  white-space: pre-wrap;
  color: #e0e0e0;
  line-height: 1.6;
}
h2 {
  text-align: center;
  margin-top: 0;
  margin-bottom: 1.5rem;
}
</style>
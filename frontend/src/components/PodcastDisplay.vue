<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
  conversationHistory: {
    type: Array,
    default: () => []
  },

  audioLog: {
    type: Object,
    default: () => ({})
  }, 

  currentSources: {
    type: Array,
    default: () => []
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
  return props.conversationHistory.map((cleanEntry, index) => {
    const audioFile = props.audioLog[index] || null;

    const match = cleanEntry.match(/^([\w\s]+):\s(.*)/s);

    let speaker = "System";
    let message = cleanEntry;

    if (match) {
        speaker = match[1].trim();
        message = match[2].trim();
    } else if (cleanEntry.startsWith("User Doubt:")) {
        speaker = "Your Question";
        message = cleanEntry.replace("User Doubt: ", "");
    }

    return { speaker, message, audioFile };
  });
});

const createSnippet = (text, maxLength = 120) => {
    if (text.length <= maxLength) {
        return text;
    }
    const trimmedText = text.slice(0, maxLength);
    return trimmedText.slice(0, trimmedText.lastIndexOf(' ')) + '...';
};

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
  <div class="podcast-display-container">
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

    <div class="sources-container" v-if="currentSources.length > 0">
      <h4 class="sources-title">Sources:</h4>
      <div class="source-icons-list">
        <div v-for="(source, index) in currentSources" :key="index" class="source-icon-wrapper">
          <span class="source-reference-tag">[ {{ index + 1 }} ]</span>
          <div class="source-tooltip">
            {{ createSnippet(source) }}
          </div>
        </div>
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
.turn.example-eve .speaker, .turn.analogy-alex .speaker, .turn.factual-finn .speaker, .turn.factual-finn-\(addressing-doubt\) .speaker { 
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

.sources-container {
    margin-top: 1rem;
    padding: 10px 25px;
    background-color: #2f2f2f;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sources-title {
    margin: 0;
    color: #ccc;
    font-weight: 600;
    font-size: 0.9em;
}
.source-icons-list {
    display: flex;
    align-items: center;
    gap: 8px;
}
.source-icon-wrapper {
    position: relative; /* Anchor for the tooltip */
    cursor: pointer;
}
.source-reference-tag {
    display: inline-block;
    padding: 2px 6px;
    background-color: #444;
    color: #00aaff;
    font-size: 0.8em;
    font-weight: bold;
    border-radius: 4px;
    border: 1px solid #555;
    transition: background-color 0.2s;
}
.source-icon-wrapper:hover .source-reference-tag {
    background-color: #007bff;
    color: white;
}
.source-tooltip {
    visibility: hidden;
    opacity: 0;
    width: 350px; /* Fixed width for snippet tooltip */
    background-color: #1e1e1e;
    color: #e0e0e0;
    text-align: left;
    border-radius: 6px;
    padding: 12px;
    position: absolute;
    z-index: 10;
    bottom: 125%; /* Position above the icon */
    left: 50%;
    transform: translateX(-50%);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    border: 1px solid #555;
    transition: opacity 0.3s ease-in-out;
    font-size: 0.9em;
    line-height: 1.6;
}

.source-tooltip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #1e1e1e transparent transparent transparent;
}

.source-icon-wrapper:hover .source-tooltip {
    visibility: visible;
    opacity: 1;
}
</style>
<script setup>
import { defineProps, computed } from 'vue';

const props = defineProps({
  conversationHistory: {
    type: Array,
    default: () => []
  }
});

// Optional: A computed property to make it easier to display speakers
const formattedHistory = computed(() => {
  return props.conversationHistory.map(entry => {
    let speaker = "Unknown";
    let message = entry;
    if (entry.startsWith("Initial Topic:")) {
      speaker = "Topic";
      message = entry.replace("Initial Topic: ", "");
    } else if (entry.startsWith("Curious Casey:")) {
      speaker = "Curious Casey";
      message = entry.replace("Curious Casey: ", "");
    } else if (entry.startsWith("Factual Finn (addressing doubt):")) {
      speaker = "Factual Finn (Doubt)";
      message = entry.replace("Factual Finn (addressing doubt): ", "");
    } else if (entry.startsWith("Factual Finn:")) {
      speaker = "Factual Finn";
      message = entry.replace("Factual Finn: ", "");
    } else if (entry.startsWith("User Doubt:")) {
      speaker = "Your Question";
      message = entry.replace("User Doubt: ", "");
    }
    return { speaker, message };
  });
});
</script>

<template>
  <div class="podcast-display" v-if="formattedHistory.length > 0">
    <h2>Podcast Conversation</h2>
    <div class="conversation-log">
      <div v-for="(turn, index) in formattedHistory" :key="index" class="turn">
        <strong class="speaker">{{ turn.speaker }}:</strong>
        <p class="message">{{ turn.message }}</p>
      </div>
    </div>
  </div>
  <div v-else class="podcast-display">
    <p>No podcast content generated yet. Fill out the form above to start!</p>
  </div>
</template>

<style scoped>
.podcast-display {
  margin-top: 20px;
  padding: 20px;
  background-color: #2f2f2f;
  border-radius: 8px;
  text-align: left;
}
.conversation-log .turn {
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid #444;
}
.conversation-log .turn:last-child {
  border-bottom: none;
  margin-bottom: 0;
  padding-bottom: 0;
}
.speaker {
  color: #00aaff; /* Example speaker color */
  margin-right: 5px;
}
.message {
  margin-top: 5px;
  margin-bottom: 0;
  white-space: pre-wrap; /* Preserve line breaks from the AI */
  color: #e0e0e0;
}
h2 {
  text-align: center;
  margin-bottom: 15px;
}
</style>
<script setup>
import { computed } from 'vue';

const props = defineProps({
  conversationHistory: {
    type: Array,
    default: () => []
  }
});

const formattedHistory = computed(() => {
  if (!props.conversationHistory || props.conversationHistory.length === 0) {
    return [];
  }
  return props.conversationHistory.map(entry => {
    let speaker = "System";
    let message = entry;
    if (entry.startsWith("Initial Topic:")) {
      speaker = "Topic";
      message = entry.replace("Initial Topic: ", "");
    } else if (entry.startsWith("Curious Casey:")) {
      speaker = "Curious Casey";
      message = entry.replace("Curious Casey: ", "");
    } else if (entry.startsWith("Factual Finn (addressing doubt):")) {
      speaker = "Factual Finn";
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
      <div v-for="(turn, index) in formattedHistory" :key="index" :class="['turn', turn.speaker.toLowerCase().replace(/\s+/g, '-')]">
        <strong class="speaker">{{ turn.speaker }}:</strong>
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
.speaker {
  font-weight: bold;
  margin-right: 8px;
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
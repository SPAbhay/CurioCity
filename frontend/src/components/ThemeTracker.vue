<script setup>
import { defineProps, computed } from 'vue';

const props = defineProps({
  guidingThemes: {
    type: Array,
    default: () => []
  },
  coveredThemes: {
    type: Array,
    default: () => []
  }
});

const isThemeCovered = (theme) => {
  return props.coveredThemes.includes(theme);
};
</script>

<template>
  <div class="theme-tracker-container" v-if="guidingThemes.length > 0">
    <h3>Podcast Guiding Themes</h3>
    <ul class="theme-list">
      <li 
        v-for="(theme, index) in guidingThemes" 
        :key="index"
        :class="{ 'covered': isThemeCovered(theme) }"
        class="theme-item"
      >
        <span class="status-icon">
          {{ isThemeCovered(theme) ? '✓' : '○' }}
        </span>
        <span class="theme-text">{{ theme }}</span>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.theme-tracker-container {
  background-color: #2f2f2f;
  padding: 20px;
  border-radius: 8px;
  margin: 20px 0;
}

h3 {
  margin-top: 0;
  text-align: center;
  border-bottom: 1px solid #444;
  padding-bottom: 10px;
  margin-bottom: 15px;
}

.theme-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.theme-item {
  display: flex;
  align-items: center;
  padding: 8px 0;
  font-size: 0.95em;
  transition: color 0.3s, text-decoration 0.3s;
  color: #e0e0e0;
}

.theme-item.covered {
  color: #777;
  text-decoration: line-through;
}

.status-icon {
  margin-right: 12px;
  font-weight: bold;
  font-size: 1.2em;
  min-width: 20px;
  text-align: center;
}

.theme-item.covered .status-icon {
  color: #28a745; 
}

.theme-item:not(.covered) .status-icon {
    color: #aaa;
}

.theme-text {
  line-height: 1.4;
}
</style>
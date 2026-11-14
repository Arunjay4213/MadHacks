import React from 'react';
import { createRoot } from 'react-dom/client';
import App from '../../src/App';
import '../../src/styles.css';

const mode = new URLSearchParams(location.search).get('mode');
const source = document.createElement('canvas');
source.width = 800; source.height = 400;
const context = source.getContext('2d')!;
context.fillStyle = '#ff0000'; context.fillRect(0, 0, 800, 400);
const mutations: string[] = [];
Object.assign(window, { mutations });
const adapter = { sendMutation: async (payload: { prompt: string }) => { mutations.push(payload.prompt); } };
createRoot(document.getElementById('root')!).render(mode === 'external'
  ? <App source={source} initialPrompt="A red test feed." /> : <App adapter={adapter} />);

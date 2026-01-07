/** @type {import('tailwindcss').Config} */
module.exports = {
  content: {
    files: ["*.html", "./src/**/*.rs"],
  },
  theme: {
    extend: {
      colors: {
        "obsidian-bg": "#1a1b1e",
        "obsidian-sidebar": "#25262b",
        "obsidian-accent": "#5c7cfa",
        "obsidian-text": "#c1c2c5",
        "obsidian-heading": "#ffffff",
      },
    },
  },
  plugins: [],
}

# ── Frontend Dockerfile ───────────────────────────────────
# Nginx serving the static UI for the Salary Predictor
# Build: docker build -t salary-ui ./frontend
# Run:   docker run -p 80:80 salary-ui

FROM nginx:1.25-alpine

# Remove default Nginx welcome page
RUN rm -rf /usr/share/nginx/html/*

# Copy our UI
COPY index.html /usr/share/nginx/html/index.html

# Copy custom Nginx config (handles SPA routing + proxy to backend)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost:80 || exit 1

CMD ["nginx", "-g", "daemon off;"]

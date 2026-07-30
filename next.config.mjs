/** @type {import('next').NextConfig} */
const nextConfig = {
  distDir: process.env.VERCEL ? ".next" : "dist",
  output: "standalone",
  poweredByHeader: false,
};

export default nextConfig;

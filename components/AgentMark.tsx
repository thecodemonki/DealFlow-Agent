type Props = { size?: number; className?: string };

export function AgentMark({ size = 44, className }: Props) {
  return (
    <svg
      width={size}
      height={(size * 195) / 260}
      viewBox="0 0 260 195"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <polygon points="41,49 117,49 114,60 38,60" fill="#ac63f0" />
      <polygon points="28,73 106,73 103,84 25,84" fill="#4084d9" />
      <polygon points="11,98 93,98 90,109 8,109" fill="#eba718" />
      <polygon points="102,124 189,124 186,135 99,135" fill="#1bad7c" />
      <polygon points="29,151 70,151 67,162 26,162" fill="#dc7725" />
      <polygon points="159,151 201,151 198,162 156,162" fill="#dc7725" />
      <polygon points="12,175 55,175 52,186 9,186" fill="#e64f44" />
      <polygon points="174,175 217,175 214,186 171,186" fill="#e64f44" />
      <path d="M87 45 L112 8 L250 124" stroke="#1e1f4d" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

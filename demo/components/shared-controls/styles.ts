export const sectionHeadingSx = {
  fontFamily: 'mono',
  letterSpacing: 'smallcaps',
  textTransform: 'uppercase' as const,
  fontSize: [2],
}

export const subheadingSx = {
  ...sectionHeadingSx,
  my: 1,
  fontSize: [1, 1, 1, 2],
  color: 'secondary',
}

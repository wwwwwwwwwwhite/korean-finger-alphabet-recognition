const ONSETS = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"];
const MEDIALS = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"];
const FINALS = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"];
const COMPOUND_MEDIALS: Record<string, string> = {
  "ㅗㅏ": "ㅘ",
  "ㅗㅐ": "ㅙ",
  "ㅗㅣ": "ㅚ",
  "ㅜㅓ": "ㅝ",
  "ㅜㅔ": "ㅞ",
  "ㅜㅣ": "ㅟ",
  "ㅡㅣ": "ㅢ",
};

function isVowel(value: string | undefined): value is string {
  return value !== undefined && MEDIALS.includes(value);
}

export function composeHangul(tokens: string[]): string {
  const result: string[] = [];
  let i = 0;
  while (i < tokens.length) {
    let onsetIndex = ONSETS.indexOf(tokens[i]);
    if (onsetIndex < 0) {
      if (isVowel(tokens[i])) {
        onsetIndex = ONSETS.indexOf("ㅇ");
      } else {
        result.push(tokens[i]);
        i += 1;
        continue;
      }
    } else {
      i += 1;
    }

    let medialJamo = tokens[i];
    if (!isVowel(medialJamo)) {
      result.push(ONSETS[onsetIndex]);
      continue;
    }
    i += 1;
    const combined = COMPOUND_MEDIALS[medialJamo + (tokens[i] ?? "")];
    if (combined) {
      medialJamo = combined;
      i += 1;
    }

    const medialIndex = MEDIALS.indexOf(medialJamo);
    let finalIndex = 0;
    const possibleFinal = FINALS.indexOf(tokens[i] ?? "");
    if (possibleFinal > 0 && !isVowel(tokens[i + 1])) {
      finalIndex = possibleFinal;
      i += 1;
    }

    result.push(String.fromCharCode(0xac00 + (onsetIndex * 21 + medialIndex) * 28 + finalIndex));
  }
  return result.join("");
}

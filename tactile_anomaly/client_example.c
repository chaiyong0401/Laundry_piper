/**
 * tactile_anomaly C client example (for xela_bridge.py)
 *
 * Protocol:
 *   1. Send 1-byte command to Server:
 *      0x01 = CMD_RESET (Initialize buffer / Start new episode)
 *      0x02 = CMD_GET   (Query current inference result)
 *
 *   2. If CMD_GET is sent, Server responds with 12 bytes:
 *      - [0]    uint8_t anomaly  (0: normal, 1: anomaly)
 *      - [1]    uint8_t channel  (0: none, 1: A, 2: B, 3: AB)
 *      - [2-5]  float   score_a  (Channel A reconstruction error)
 *      - [6-9]  float   delta_b  (Channel B frame-wise MRL delta)
 *      - [10]   uint8_t state    (0: IDLE, 1: NORMAL, 2: ANOMALY)
 *      - [11]   uint8_t reserved (0 padding)
 *
 * Build:
 *   gcc -o client client_example.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <unistd.h>

#define SERVER_IP   "127.0.0.1"
#define SERVER_PORT 9001
#define RESULT_BYTES 12

#define CMD_RESET 0x01
#define CMD_GET   0x02

typedef struct {
    uint8_t anomaly;   /* 0 or 1 */
    uint8_t channel;   /* 0=none, 1=A, 2=B, 3=AB */
    float   score_a;
    float   delta_b;
    uint8_t state;     /* 0=IDLE, 1=NORMAL, 2=ANOMALY */
    uint8_t reserved;
} TactileResult;

static const char *channel_name[] = {"none", "A", "B", "AB"};
static const char *state_name[] = {"IDLE", "NORMAL", "ANOMALY"};

int send_all(int fd, const void *buf, size_t len) {
    const char *p = buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return -1;
        p += n; len -= n;
    }
    return 0;
}

int recv_all(int fd, void *buf, size_t len) {
    char *p = buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return -1;
        p += n; len -= n;
    }
    return 0;
}

int main(void) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return 1; }

    struct sockaddr_in srv = {0};
    srv.sin_family      = AF_INET;
    srv.sin_port        = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &srv.sin_addr);

    if (connect(sock, (struct sockaddr *)&srv, sizeof(srv)) < 0) {
        perror("connect");
        close(sock);
        return 1;
    }
    printf("[Client] xela_bridge 서버 연결 성공 (%s:%d)\n", SERVER_IP, SERVER_PORT);

    uint8_t reset_cmd = CMD_RESET;
    if (send_all(sock, &reset_cmd, 1) < 0) {
        fprintf(stderr, "[Client] RESET 명령 전송 실패\n");
        close(sock);
        return 1;
    }
    printf("[Client] RESET 명령 전송 완료 (추론 버퍼 초기화)\n");


    printf("시작 시간  |  state  |  ch  |  anomaly  |  score_A  |  delta_B\n");
    printf("-------------------------------------------------------------\n");
    for (int i = 0; i < 100; i++) {
        usleep(100000); // 100ms 대기 (10Hz)

        uint8_t get_cmd = CMD_GET;
        if (send_all(sock, &get_cmd, 1) < 0) {
            fprintf(stderr, "[Client] GET 명령 전송 실패\n");
            break;
        }

        uint8_t buf[RESULT_BYTES];
        if (recv_all(sock, buf, RESULT_BYTES) < 0) {
            fprintf(stderr, "[Client] 결과 수신 실패\n");
            break;
        }

        TactileResult res;
        res.anomaly = buf[0];
        res.channel = buf[1];
        memcpy(&res.score_a, &buf[2], 4);
        memcpy(&res.delta_b, &buf[6], 4);
        res.state = buf[10];
        res.reserved = buf[11];

        const char *ch_str = channel_name[res.channel < 4 ? res.channel : 0];
        const char *st_str = state_name[res.state < 3 ? res.state : 0];

        printf("query %3d  | %-7s | %-4s |     %d     |  %8.4f |  %7.4f\n",
               i, st_str, ch_str, res.anomaly, res.score_a, res.delta_b);
    }

    close(sock);
    return 0;
}

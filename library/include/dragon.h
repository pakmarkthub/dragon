#ifndef __DRAGON_HEADER__
#define __DRAGON_HEADER__

#include <stdio.h>

/* Flags for dragon_map */
#define D_F_READ        0x01
#define D_F_WRITE       0x02
#define D_F_CREATE      0x04
#define D_F_DONTTRASH   0x08
#define D_F_VOLATILE    0x10
#define D_F_USEHOSTBUF  0x20

/* Errors */
typedef enum {
    D_OK        = 0,
    D_ERR_FILE,
    D_ERR_IOCTL,
    D_ERR_UVM,
    D_ERR_INTVAL,
    D_ERR_MEM,
    D_ERR_NOT_IMPLEMENTED
} dragonError_t;

#ifdef __cplusplus
extern "C"
{
#endif
    dragonError_t dragon_map(const char *filename, size_t size, unsigned short flags, void **addr);
    dragonError_t dragon_remap(void *addr, unsigned short flags);
    dragonError_t dragon_trash_set_num_blocks(unsigned long nrblocks);
    dragonError_t dragon_trash_set_num_reserved_sys_cache_pages(unsigned long nrpages);
    dragonError_t dragon_flush(void *addr);
    dragonError_t dragon_unmap(void *addr);
#ifdef __cplusplus
}
#endif

#endif

